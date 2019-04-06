import torch
import sys
import numpy as np
import itertools
from models import *
from dataset import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import time
import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/UCF-101-frames", help="Path to UCF-101 dataset")
    parser.add_argument("--split_path", type=str, default="data/ucfTrainTestlist", help="Path to train/test split")
    parser.add_argument("--split_number", type=int, default=1, help="train/test split number. One of {1, 2, 3}")
    parser.add_argument("--img_dim", type=int, default=112, help="Height / width dimension")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--latent_dim", type=int, default=512, help="Dimensionality of the latent representation")
    parser.add_argument("--checkpoint_model", type=str, help="Optional path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    assert opt.checkpoint_model, "Specify path to checkpoint model using arg. '--checkpoint_model'"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_shape = (opt.channels, opt.img_dim, opt.img_dim)

    # Define test set
    test_dataset = Dataset(
        dataset_path=opt.dataset_path,
        split_path=opt.split_path,
        split_number=opt.split_number,
        input_shape=image_shape,
        sequence_length=None,
        training=False,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Define network
    model = ConvLSTM(
        num_classes=train_dataset.num_classes,
        latent_dim=opt.latent_dim,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
    )
    model = model.to(device)
    model.load_state_dict(torch.load(opt.checkpoint_model))
    model.eval()

    test_accuracies = []
    for batch_i, (X, y) in enumerate(test_dataloader):
        image_sequences = Variable(X.to(device), requires_grad=False)
        labels = Variable(y, requires_grad=False).to(device)
        with torch.no_grad():
            # Reset LSTM hidden state
            model.lstm.reset_hidden_state()
            # Get sequence predictions
            predictions = model(image_sequences)
            loss = 0
            # Update loss and prediction histogram at each timestep
            pred_hists = np.zeros((predictions.size(0), predictions.size(-1)))
            for t in range(opt.sequence_length):
                # Update classification loss
                loss += cls_criterion(predictions[:, t], labels).item() / opt.sequence_length
                # Update prediction histogram
                pred_hists[:, predictions[:, t].argmax(1).cpu().numpy()] += 1
        # Compute accuracy using the most common prediction for each sequence
        acc = 100 * np.mean(pred_hists.argmax(1) == labels.cpu().numpy())
        # Keep track of accuracy
        test_accuracies.append(acc)
        # Log test performance
        sys.stdout.write(
            "\rTesting -- [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)]"
            % (batch_i, len(test_dataloader), loss, np.mean(test_metrics["loss"]), acc, np.mean(test_accuracies))
        )
