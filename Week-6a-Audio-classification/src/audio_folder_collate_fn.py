import torch 
import numpy as np

# These functions are originally from: https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html
def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def collate_audio_folder_batch(batch):
    # Empty lists for adding data too
    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, label in batch:
        tensors += [torch.Tensor(waveform)]
        targets += [label]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    # tensors = tensors.squeeze(1)
    targets = np.stack(targets)
    targets = torch.Tensor(targets)
    targets = targets.to(torch.long)
    return tensors, targets