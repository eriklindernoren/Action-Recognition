from models import *
from dataset import *
from data.extract_frames import extract_frames
import argparse
import os
import glob
import tqdm
from torchvision.utils import make_grid
from PIL import Image, ImageDraw
import skvideo.io

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path", type=str, default="data/UCF-101/BabyCrawling/v_BabyCrawling_g01_c01.avi", help="Path to video"
    )
    parser.add_argument("--dataset_path", type=str, default="data/UCF-101-frames", help="Path to UCF-101 dataset")
    parser.add_argument("--image_dim", type=int, default=112, help="Height / width dimension")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--latent_dim", type=int, default=512, help="Dimensionality of the latent representation")
    parser.add_argument("--checkpoint_model", type=str, default="", help="Optional path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    assert opt.checkpoint_model, "Specify path to checkpoint model using arg. '--checkpoint_model'"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (opt.channels, opt.image_dim, opt.image_dim)

    transform = transforms.Compose(
        [
            transforms.Resize(input_shape[-2:], Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    labels = sorted(list(set(os.listdir(opt.dataset_path))))

    # Define model and load model checkpoint
    model = ConvLSTM(input_shape=input_shape, num_classes=len(labels), latent_dim=opt.latent_dim)
    model.to(device)
    model.load_state_dict(torch.load(opt.checkpoint_model))
    model.eval()

    # Extract predictions
    output_frames = []
    for frame in tqdm.tqdm(extract_frames(opt.video_path,0), desc="Processing frames"):
        image_tensor = Variable(transform(frame)).to(device)
        image_tensor = image_tensor.view(1, 1, *image_tensor.shape)

        # Get label prediction for frame
        with torch.no_grad():
            prediction = model(image_tensor)
            predicted_label = labels[prediction.argmax(1).item()]

        # Draw label on frame
        d = ImageDraw.Draw(frame)
        d.text(xy=(10, 10), text=predicted_label, fill=(255, 255, 255))

        output_frames += [frame]

    # Create video from frames
    writer = skvideo.io.FFmpegWriter("output.gif")
    for frame in tqdm.tqdm(output_frames, desc="Writing to video"):
        writer.writeFrame(np.array(frame))
    writer.close()
