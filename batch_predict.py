"""
Batch Prediction Script for Plant Disease Classification

This script allows you to process multiple images at once and generate
a comprehensive report of all predictions.
"""

import os
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import argparse
from utils.predict import PlantDiseasePredictor
from typing import List, Dict

class BatchPredictor:
    """Batch prediction class for processing multiple images."""
    
    def __init__(self, model_path: str = "model/Plant_Disease_Detection.h5", 
                 class_indices_path: str = "assets/class_indices.json"):
        """
        Initialize the batch predictor.
        
        Args:
            model_path: Path to the trained model
            class_indices_path: Path to class indices JSON file
        """
        self.predictor = PlantDiseasePredictor(model_path, class_indices_path)
        self.results = []
    
    def predict_folder(self, folder_path: str, output_dir: str = "batch_results") -> List[Dict]:
        """
        Process all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            output_dir: Directory to save results
            
        Returns:
            List of prediction results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(folder_path).glob(f"*{ext}"))
            image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No image files found in {folder_path}")
            return []
        
        print(f"Found {len(image_files)} images to process...")
        
        results = []
        for i, image_path in enumerate(image_files):
            try:
                print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
                
                # Make prediction
                class_name, confidence, prediction_details = self.predictor.predict_disease(str(image_path))
                disease_info = self.predictor.get_disease_info(class_name)
                
                # Store result
                result = {
                    'filename': image_path.name,
                    'filepath': str(image_path),
                    'predicted_class': class_name,
                    'confidence': confidence,
                    'confidence_percentage': confidence * 100,
                    'plant_type': disease_info['plant_type'],
                    'condition': disease_info['condition'],
                    'is_healthy': disease_info['is_healthy'],
                    'formatted_name': disease_info['formatted_name'],
                    'timestamp': datetime.now().isoformat(),
                    'top_3_predictions': prediction_details['top_predictions']
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {image_path.name}: {str(e)}")
                results.append({
                    'filename': image_path.name,
                    'filepath': str(image_path),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Save results
        self.save_results(results, output_dir)
        
        return results
    
    def save_results(self, results: List[Dict], output_dir: str):
        """
        Save results to various formats.
        
        Args:
            results: List of prediction results
            output_dir: Directory to save results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_path = os.path.join(output_dir, f"predictions_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {json_path}")
        
        # Save as CSV
        csv_path = os.path.join(output_dir, f"predictions_{timestamp}.csv")
        
        # Prepare data for CSV (flatten top_3_predictions)
        csv_data = []
        for result in results:
            if 'error' not in result:
                row = {
                    'filename': result['filename'],
                    'predicted_class': result['predicted_class'],
                    'confidence_percentage': result['confidence_percentage'],
                    'plant_type': result['plant_type'],
                    'condition': result['condition'],
                    'is_healthy': result['is_healthy'],
                    'formatted_name': result['formatted_name'],
                    'timestamp': result['timestamp']
                }
                
                # Add top 3 predictions
                if 'top_3_predictions' in result:
                    top_preds = result['top_3_predictions']
                    for i, (pred_name, pred_conf) in enumerate(top_preds.items()):
                        row[f'top_{i+1}_prediction'] = pred_name
                        row[f'top_{i+1}_confidence'] = pred_conf * 100
                
                csv_data.append(row)
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False)
            print(f"CSV results saved to: {csv_path}")
        
        # Generate summary report
        self.generate_summary_report(results, output_dir, timestamp)
    
    def generate_summary_report(self, results: List[Dict], output_dir: str, timestamp: str):
        """
        Generate a summary report.
        
        Args:
            results: List of prediction results
            output_dir: Directory to save results
            timestamp: Timestamp for filename
        """
        # Filter out errors
        valid_results = [r for r in results if 'error' not in r]
        error_results = [r for r in results if 'error' in r]
        
        # Calculate statistics
        total_images = len(results)
        successful_predictions = len(valid_results)
        failed_predictions = len(error_results)
        
        # Disease distribution
        disease_counts = {}
        healthy_count = 0
        plant_type_counts = {}
        
        for result in valid_results:
            condition = result['condition']
            plant_type = result['plant_type']
            
            if result['is_healthy']:
                healthy_count += 1
            
            disease_counts[condition] = disease_counts.get(condition, 0) + 1
            plant_type_counts[plant_type] = plant_type_counts.get(plant_type, 0) + 1
        
        # Average confidence
        if valid_results:
            avg_confidence = sum(r['confidence_percentage'] for r in valid_results) / len(valid_results)
        else:
            avg_confidence = 0
        
        # Generate report
        report = f"""# Plant Disease Classification Batch Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics
- **Total Images Processed:** {total_images}
- **Successful Predictions:** {successful_predictions}
- **Failed Predictions:** {failed_predictions}
- **Success Rate:** {(successful_predictions/total_images*100):.1f}%
- **Average Confidence:** {avg_confidence:.1f}%

## Health Status Distribution
- **Healthy Plants:** {healthy_count} ({healthy_count/successful_predictions*100:.1f}% if successful_predictions > 0 else 0)
- **Diseased Plants:** {successful_predictions - healthy_count} ({(successful_predictions - healthy_count)/successful_predictions*100:.1f}% if successful_predictions > 0 else 0)

## Plant Type Distribution
"""
        
        for plant_type, count in sorted(plant_type_counts.items()):
            percentage = (count / successful_predictions * 100) if successful_predictions > 0 else 0
            report += f"- **{plant_type}:** {count} ({percentage:.1f}%)\n"
        
        report += "\n## Disease/Condition Distribution\n"
        for condition, count in sorted(disease_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / successful_predictions * 100) if successful_predictions > 0 else 0
            report += f"- **{condition}:** {count} ({percentage:.1f}%)\n"
        
        if error_results:
            report += "\n## Failed Predictions\n"
            for error_result in error_results:
                report += f"- **{error_result['filename']}:** {error_result['error']}\n"
        
        # Save report
        report_path = os.path.join(output_dir, f"summary_report_{timestamp}.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Summary report saved to: {report_path}")
        print("\n" + "="*50)
        print("BATCH PROCESSING SUMMARY")
        print("="*50)
        print(f"Total images: {total_images}")
        print(f"Successful: {successful_predictions}")
        print(f"Failed: {failed_predictions}")
        print(f"Average confidence: {avg_confidence:.1f}%")
        print(f"Healthy plants: {healthy_count}")
        print(f"Diseased plants: {successful_predictions - healthy_count}")

def main():
    """Main function for batch prediction."""
    parser = argparse.ArgumentParser(description="Batch Plant Disease Prediction")
    parser.add_argument("folder_path", help="Path to folder containing images")
    parser.add_argument("--output", "-o", default="batch_results", help="Output directory for results")
    parser.add_argument("--model", "-m", default="model/Plant_Disease_Detection.h5", help="Path to model file")
    parser.add_argument("--classes", "-c", default="assets/class_indices.json", help="Path to class indices file")
    
    args = parser.parse_args()
    
    # Check if folder exists
    if not os.path.exists(args.folder_path):
        print(f"Error: Folder '{args.folder_path}' does not exist!")
        return
    
    # Check if model files exist
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' does not exist!")
        return
    
    if not os.path.exists(args.classes):
        print(f"Error: Class indices file '{args.classes}' does not exist!")
        return
    
    print("Plant Disease Batch Prediction")
    print("=" * 40)
    print(f"Input folder: {args.folder_path}")
    print(f"Output directory: {args.output}")
    print(f"Model: {args.model}")
    print(f"Class indices: {args.classes}")
    print("=" * 40)
    
    # Initialize batch predictor
    batch_predictor = BatchPredictor(args.model, args.classes)
    
    # Process folder
    results = batch_predictor.predict_folder(args.folder_path, args.output)
    
    print(f"\nBatch processing completed! Results saved to '{args.output}' directory.")

if __name__ == "__main__":
    # If no command line arguments, use example
    import sys
    if len(sys.argv) == 1:
        print("Usage: python batch_predict.py <folder_path> [--output output_dir]")
        print("\nExample:")
        print("python batch_predict.py test_images --output results")
        print("\nTo process images in 'test_images' folder and save results to 'results' directory")
    else:
        main()
