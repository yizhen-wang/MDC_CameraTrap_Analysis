import os
import torch
import argparse
import detection as det
import classification as cla
import engine as engine

def wildlife_pipeline(data_dir, **kwargs):
    """
    Entry function for the wildlife analysis pipeline.

    Parameters:
    - data_dir (str): Required. Directory containing input images.
    - kwargs: Optional parameter dictionary including:
        - output_dir (str): Directory to save results, defaults to the current folder.
        - seq (bool): Whether to process images in sequence mode, defaults to False.
        - conf_threshold (float): Confidence threshold for the detection step, default is 0.5.
        - motion_analysis (bool): Whether to use motion-based algorithms to enhance detection results, defaults to False.
        - save_format (str): Format to save results, default is "json", alternative is "csv".
        - save_images (bool): Whether to save images with bounding boxes and classification results, defaults to False.
    """
    # Print pipeline configuration
    print("Running Wildlife Analysis Pipeline with the following settings:")
    print(f"Data Directory: {data_dir}")
    print(f"Output Directory: {kwargs.get('output_dir')}")
    print(f"Sequence Mode: {kwargs.get('seq')}")
    print(f"Confidence Threshold: {kwargs.get('conf_threshold')}")
    print(f"Motion Analysis: {kwargs.get('motion_analysis')}")
    print(f"Save Format: {kwargs.get('save_format')}")
    print(f"Save Images: {kwargs.get('save_images')}")

    output_dir = kwargs.get('output_dir')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(0)

    animal_detector   = det.Animal_Detector(device)
    animal_classifier = cla.Animal_Classifier(device)
    animal_analyser   = engine.Animal_Detector(data_dir, output_dir, animal_detector, animal_classifier)
    animal_analyser.run_analysis()



    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wildlife Analysis Pipeline")

    # Add required parameter
    parser.add_argument("data_dir", type=str, help="Directory containing input images")

    # Add optional parameters
    parser.add_argument("--output_dir", type=str, default=os.getcwd(), 
                        help="Directory to save results (default: current folder)")
    parser.add_argument("--seq", action="store_true", default=False, 
                        help="Process images in sequence mode (default: False)")
    parser.add_argument("--conf_threshold", type=float, default=0.5, 
                        help="Confidence threshold for detection (default: 0.5)")
    parser.add_argument("--motion_analysis", action="store_true", default=False, 
                        help="Enable motion-based algorithm to enhance detection (default: False)")
    parser.add_argument("--save_format", type=str, choices=["json", "csv"], default="json", 
                        help="Format to save results (default: json)")
    parser.add_argument("--save_images", action="store_true", default=False, 
                        help="Save images with bounding boxes and classification results (default: False)")

    args = parser.parse_args()
    kwargs = vars(args)
    wildlife_pipeline(**kwargs)


