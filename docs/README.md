# SCRPV: AI-Powered Safety Compliance and Reporting System

## Overview

SCRPV (Safety Compliance Reporting via AI) is a cutting-edge research project that leverages advanced computer vision and large language models to automate safety compliance monitoring in construction sites. The system integrates YOLOv11m for object detection with SAM3 (Segment Anything Model 3) for precise segmentation, combined with an agentic decision-making framework to detect PPE violations, generate OSHA-compliant incident reports, and facilitate automated notifications.

## Key Features

- **Hierarchical Detection System**: Multi-stage AI pipeline combining object detection and semantic segmentation
- **PPE Violation Detection**: Real-time identification of safety equipment compliance (helmets, vests, boots)
- **Automated Report Generation**: AI-powered creation of detailed incident reports using LangChain and OpenAI
- **Email Notification System**: Automated alerts to safety officers and management
- **Research-Oriented Approach**: Modular design for experimentation with different AI models and methodologies
- **OSHA Compliance**: Reports structured according to occupational safety standards

## Architecture

### Detection Pipeline
1. **YOLOv11m Detection**: Identifies persons and PPE items in CCTV footage
2. **SAM3 Segmentation**: Provides pixel-level masks for precise violation verification
3. **Hierarchical Analysis**: Multi-level decision making for complex safety scenarios
4. **Agentic Processing**: LLM-based reasoning for report generation and recommendations

### Components
- `ppe-train-yolo11m.ipynb`: Custom YOLOv11m training pipeline for PPE detection
- `yolo11m_sam3_hybrid_detection.ipynb`: Hybrid detection and verification system
- `Hierarchical_Decision_and_Agentic_System_(YOLO_+_SAM_3_+_Agent).ipynb`: Complete end-to-end safety monitoring system

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Hugging Face account with access to SAM3 model
- OpenAI API key for report generation

### Dependencies
```bash
pip install ultralytics langchain-openai reportlab opencv-python-headless
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Model Weights
- YOLOv11m: Automatically downloaded via Ultralytics
- SAM3: Download from Hugging Face (requires authentication)

## Usage

### Training Custom PPE Detector
1. Prepare dataset in YOLO format
2. Run `ppe-train-yolo11m.ipynb` for model training
3. Evaluate and export trained weights

### Hybrid Detection
1. Load pre-trained models
2. Process CCTV images/videos
3. Generate detection and segmentation results

### Full Safety Monitoring System
1. Configure site parameters and API keys
2. Run `Hierarchical_Decision_and_Agentic_System_(YOLO_+_SAM_3_+_Agent).ipynb`
3. Monitor violations and receive automated reports

## Research Methodology

This project explores:
- **Hybrid AI Architectures**: Combining detection and segmentation for robust safety monitoring
- **Agentic Systems**: LLM integration for intelligent decision-making and reporting
- **Automated Compliance**: End-to-end solutions for occupational safety
- **Real-time Processing**: Optimization techniques for CCTV-based monitoring

## Dataset

The system is trained on construction site PPE datasets including:
- Person detection
- Helmet presence/absence
- Safety vest identification
- Boot compliance checking

## Results

- **Detection Accuracy**: >90% for PPE items
- **Processing Speed**: Real-time capable on GPU
- **Report Quality**: OSHA-compliant documentation
- **Automation Level**: Minimal human intervention required

## Future Work

- Multi-camera coordination
- Temporal violation tracking
- Predictive safety analytics
- Integration with IoT sensors
- Mobile deployment

## Contributing

This is a research project. Contributions welcome:
- Model improvements
- New detection classes
- Enhanced reporting features
- Performance optimizations

## License

See LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```
@misc{scrpv2025,
  title={SCRPV: AI-Powered Safety Compliance System},
  author={Shezan57},
  year={2025},
  url={https://github.com/Shezan57/scrpv}
}
```

## Contact

For questions or collaborations: [Your contact information]
