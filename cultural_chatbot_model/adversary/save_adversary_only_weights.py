"""Saves the adversary weights without the other details from training"""
import torch

checkpoint = torch.load("best_checkpoint.pt", map_location=torch.device('cpu'))

model_weights = checkpoint['model_state_dict']

torch.save(model_weights, "adversary_weights_v1.pt")
