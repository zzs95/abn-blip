"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from itertools import chain 
from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.abnclip_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import ABNClipOutput, BlipOutputFeatures

@registry.register_model("abnclip")
# @registry.register_model("blip2")
# @registry.register_model("blip2_feature_extractor")
class Blip2Qformer(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/abnclip/blip2_pretrain.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=31,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.abn_size = 32
        # num_query_token = self.abn_size
        num_query_token = 1
        self.num_query_token = num_query_token
        self.visual_encoder_num_features = 1408
        self.Qformer, self.query_tokens = self.init_Qformer(
            self.num_query_token, self.visual_encoder_num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        
        max_txt_len = 40
        self.max_txt_len = max_txt_len

        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=512, out_channels=1408, kernel_size=1, stride=1, padding=0),
            nn.Flatten(start_dim=2, end_dim=4),
            )
        self.layer2 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=512, out_channels=1408, kernel_size=1, stride=1, padding=0),
            nn.Flatten(start_dim=2, end_dim=4),
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(),
            nn.Conv3d(in_channels=512, out_channels=1408, kernel_size=1, stride=1, padding=0),
            nn.Flatten(start_dim=2, end_dim=4),            
            )
        
        self.layer4 = nn.Sequential(
            nn.Conv3d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(),
            nn.Conv3d(in_channels=512, out_channels=1408, kernel_size=1, stride=1, padding=0),
            nn.Flatten(start_dim=2, end_dim=4),
            nn.Linear(2 * 2 * 2, 32)) 
        
        self.layer5 = nn.Sequential(
            nn.Conv3d(in_channels=2048, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(),
            nn.Conv3d(in_channels=512, out_channels=1408, kernel_size=1, stride=1, padding=0),
            nn.Flatten(start_dim=2, end_dim=4),
            nn.Linear(7 * 7 * 10, 256),
            nn.ReLU(),
            nn.Linear(256, 32)
        )
        
        self.cls_embedder = nn.Sequential(
            nn.Linear(in_features=32, out_features=1408, bias=True,),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(
                in_channels=2048,
                out_channels=self.abn_size,
                kernel_size=(1, 1, 1),
                bias=True,)
            )
        
        self.cls_loss_function = nn.BCEWithLogitsLoss()
        
    def _soft_xent_loss(self, sim_i2t, soft_label):
        logprobs = F.log_softmax(sim_i2t, dim=-1)
        target = F.softmax(soft_label, dim=-1)
        return  -(target * logprobs).sum() 
    
    def forward(self, samples):
        # image = samples["image"]
        feat1 = samples["feat1"]
        feat2 = samples["feat2"]
        feat3 = samples["feat3"]
        feat4 = samples["feat4"]
        feat5 = samples["feat5"]
        text = samples["abn_text_input"]
        abn_label = samples['abn_label']
        device = feat1.device
        bs = abn_label.shape[0]
        abn_size = self.abn_size
        
        abn_logits = self.classifier(feat5)
        abn_logits = abn_logits.reshape(bs, self.abn_size)
        abn_probs = torch.sigmoid(abn_logits)
        
        loss_cls = self.cls_loss_function(abn_logits, abn_label.float())
        
        image_atts = torch.ones((bs*abn_size, 257), dtype=torch.long).to(device) # [100, 257]
        feat1_embed = self.layer1(feat1).transpose(1,2) # torch.Size([40, 64, 1408])
        feat2_embed = self.layer2(feat2).transpose(1,2) # torch.Size([40, 64, 1408])
        feat3_embed = self.layer3(feat3).transpose(1,2) # torch.Size([40, 64, 1408])
        feat4_embed = self.layer4(feat4).transpose(1,2) # torch.Size([40, 32, 1408])
        feat5_embed = self.layer5(feat5).transpose(1,2) # torch.Size([40, 32, 1408])
        cls_embed = self.cls_embedder(abn_probs[:,None])
        image_embeds = torch.concat([cls_embed, feat1_embed, feat2_embed, feat3_embed, feat4_embed, feat5_embed], dim=1)
        image_embeds_a = image_embeds[:, None].expand(-1, abn_size, -1, -1).reshape(bs*abn_size, 257, 1408) # [img000,img111,img222,...]
        
        query_tokens = self.query_tokens[None].expand(bs, -1, -1, -1).reshape(bs*abn_size, self.num_query_token, 768) # [abn0abn1abn2,abn0abn1abn2,abn0abn1abn2,...]
        
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_a,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )
       
        # b, 32, 256
        image_abn_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )
        image_feats = image_abn_feats.reshape(bs, abn_size, self.num_query_token, 256)
        
        text_tokens = self.tokenizer(
            list(chain(*text)),
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)
        
        text_output = self.Qformer.bert(
            input_ids=text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
        text_feat = text_feat.reshape(bs, abn_size, 256)# .permute(1, 0, 2)
        ###============== Image-text Contrastive ===================###
        image_feats_all = concat_all_gather(image_feats)  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]
        image_feats_all = image_feats_all.permute(1,0,2,3)
        image_feats = image_feats.permute(1,0,2,3)
        text_feat_all = text_feat_all.permute(1, 0, 2)
        text_feat = text_feat.permute(1, 0, 2)
        # abnormality-wise image-text similarity in Matrix operations
        # [32, 30, 1, 10, 256] * [32, 1, 30, 256, 1] -> [32, 30, 30, 10, 1]
        sim_q2t = torch.matmul(
                image_feats.unsqueeze(2), text_feat_all.unsqueeze(1).unsqueeze(-1)
            ).squeeze() 
        # sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t= sim_q2t
        sim_i2t = sim_i2t / self.temp
        
        # # [32, 30, 1, 1, 256] * [32, 1, 30, 256, 10] -> [32, 30, 30, 10, 1]
        sim_t2q = torch.matmul(
                text_feat.unsqueeze(2).unsqueeze(2), image_feats_all.permute(0, 1, 3, 2).unsqueeze(1)
            ).squeeze() 
        # sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2q
        sim_t2i = sim_t2i / self.temp              

        # [32, 30, 1] * [32, 1, 30] -> [32, 30, 30, 1]
        abn_probs_all = concat_all_gather(abn_probs)
        abn_probs = abn_probs.permute(1, 0)
        abn_probs_all = abn_probs_all.permute(1, 0)
        sim_abn_prob = torch.matmul(abn_probs.unsqueeze(2), abn_probs_all.unsqueeze(1))
    
        loss_itc = (
            self._soft_xent_loss(sim_i2t, sim_abn_prob)/ (bs*abn_size) + 
            self._soft_xent_loss(sim_t2i, sim_abn_prob)/ (bs*abn_size)
            ) / 2
    
        # rank = dist.get_rank()
        rank = 0

        ##================= Image Captioning ========================##
        abn_labels = abn_label.reshape(-1)
        abn_index = torch.where(abn_labels)[0]
        n_index = torch.where(abn_labels==0)[0]
        n_index = n_index[torch.randint(0, len(n_index), (len(abn_index),)) ]
        cap_index = torch.concat([abn_index, n_index])
        cap_cls_label = torch.concat([torch.ones_like(abn_index), torch.zeros_like(n_index)])
        cap_choice = torch.randint(0, len(cap_index), (300,))
        cap_index = cap_index[cap_choice]
        cap_cls_label = cap_cls_label[cap_choice]
        decoder_input_ids = text_tokens.input_ids.clone()[cap_index]
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens[cap_index].size()[:-1], dtype=torch.long).to(device)
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask[cap_index]], dim=1)
        past_key_values_list = []
        for v in query_output.past_key_values:
            past_key_values_list.append((v[0][cap_index], v[1][cap_index]))
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values_list,
            return_dict=True,
            labels=labels,
        )
        loss_lm = lm_output.loss
   
        loss_text_cls = 0
        return ABNClipOutput(
            loss=loss_itc + loss_cls + loss_lm + loss_text_cls,
            loss_itc=loss_itc,
            loss_cls=loss_cls,
            loss_lm=loss_lm,
            loss_textcls=loss_text_cls,
        )

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        
        feat1 = samples["feat1"]
        feat2 = samples["feat2"]
        feat3 = samples["feat3"]
        feat4 = samples["feat4"]
        feat5 = samples["feat5"]
        device = feat1.device
        bs = feat1.shape[0]
        abn_size = self.abn_size
        
        abn_logits = self.classifier(feat5)
        abn_logits = abn_logits.reshape(bs, self.abn_size)
        abn_probs = torch.sigmoid(abn_logits)
        
        feat1_embed = self.layer1(feat1).transpose(1,2)
        feat2_embed = self.layer2(feat2).transpose(1,2)
        feat3_embed = self.layer3(feat3).transpose(1,2)
        feat4_embed = self.layer4(feat4).transpose(1,2)
        feat5_embed = self.layer5(feat5).transpose(1,2)
        cls_embed = self.cls_embedder(abn_probs[:,None])
        image_embeds = torch.concat([cls_embed, feat1_embed, feat2_embed, feat3_embed, feat4_embed, feat5_embed], dim=1)
        image_embeds_a = image_embeds[:, None].expand(-1, abn_size,  -1, -1).reshape(bs*abn_size, 257, 1408) # [img000,img111,img222,...]
        
        query_tokens = self.query_tokens[None].expand(bs, -1, -1, -1).reshape(bs*abn_size, self.num_query_token, 768) # [abn0abn1abn2,abn0abn1abn2,abn0abn1abn2,...]
        
        image_atts = torch.ones(image_embeds_a.size()[:-1], dtype=torch.long).to(device)
        
        model_kwargs = {
            "encoder_hidden_states": image_embeds_a,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(bs*abn_size, 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(device)
        )

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        captions_list = np.array(captions).reshape(bs, abn_size)
        
        captions_list = captions_list[:,None]
        abn_probs_list = abn_probs.detach().cpu().numpy().reshape(bs, 1, abn_size)
        
        captions_out = np.concatenate([captions_list, abn_probs_list], axis=1).tolist()
        return captions_out

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            feat1 = samples["feat1"]
            feat2 = samples["feat2"]
            feat3 = samples["feat3"]
            feat4 = samples["feat4"]
            feat5 = samples["feat5"]
            abn_label = samples['abn_label']
            device = feat1.device
            bs = abn_label.shape[0]
            abn_size = self.abn_size
            
            abn_logits = self.classifier(feat5)
            abn_logits = abn_logits.reshape(bs, self.abn_size)
            abn_probs = torch.sigmoid(abn_logits)
            
            image_atts = torch.ones((bs*abn_size, 257), dtype=torch.long).to(device) # [100, 257]
            feat1_embed = self.layer1(feat1).transpose(1,2) # torch.Size([40, 64, 1408])
            feat2_embed = self.layer2(feat2).transpose(1,2) # torch.Size([40, 64, 1408])
            feat3_embed = self.layer3(feat3).transpose(1,2) # torch.Size([40, 64, 1408])
            feat4_embed = self.layer4(feat4).transpose(1,2) # torch.Size([40, 32, 1408])
            feat5_embed = self.layer5(feat5).transpose(1,2) # torch.Size([40, 32, 1408])
            cls_embed = self.cls_embedder(abn_probs[:,None])
            image_embeds = torch.concat([cls_embed, feat1_embed, feat2_embed, feat3_embed, feat4_embed, feat5_embed], dim=1)
            image_embeds_a = image_embeds[:, None].expand(-1, abn_size, -1, -1).reshape(bs*abn_size, 257, 1408) # [img000,img111,img222,...]
            
            query_tokens = self.query_tokens[None].expand(bs, -1, -1, -1).reshape(bs*abn_size, self.num_query_token, 768) # [abn0abn1abn2,abn0abn1abn2,abn0abn1abn2,...]
            
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_a,
                encoder_attention_mask=image_atts,
                use_cache=True,
                return_dict=True,
            )
            
            # b, 32, 256
            image_abn_feats = F.normalize(
                self.vision_proj(query_output.last_hidden_state), dim=-1
            )
            image_features = image_abn_feats.reshape(bs, abn_size, self.num_query_token, 256)

            image_embeds = query_output.last_hidden_state
            
        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            text_tokens = self.tokenizer(
                list(chain(*caption)),
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)                     
            text_output = self.Qformer.bert(
                input_ids=text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state[:, 0, :]
            text_feat = F.normalize(
                self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            )
            text_features = text_feat.reshape(self.abn_size, 256)# .permute(1, 0, 2)
            
        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
