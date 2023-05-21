import torch
from semdiffusers import SemanticEditPipeline


class CustomSemanticEditPipeline(SemanticEditPipeline):
    @torch.no_grad()
    def pred_noise(self, images, prompt, split_size=5):
        noise_pred = []
        for _images in images.split(split_size):
            _noise_pred = self._pred_noise(_images, prompt)
            noise_pred.append(_noise_pred)
        return torch.cat(noise_pred)

    @torch.no_grad()
    def _pred_noise(self, images, prompt):
        # encode images
        posterior = self.vae.encode(images).latent_dist
        latents = posterior.mode()

        # encode prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
        text_embeddings = text_embeddings.repeat(images.shape[0], 1, 1)

        # compute score
        t = 0  # assume timestep is 0
        latent_model_input = latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        return noise_pred
