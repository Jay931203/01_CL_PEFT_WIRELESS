---
tags:
- transformers
- wireless-communication
- zero-shot-classification
- limited-data
- feature-extraction
- pytorch
datasets:
- DeepMIMO
base_model:
- wi-lab/lwm
---

# **LWM-v1.1**

**[🚀 Click here to try the Interactive Demo Based on LWM-v1.0!](https://huggingface.co/spaces/wi-lab/lwm-interactive-demo)**

LWM-v1.1 is a powerful **pre-trained** model developed as a **universal feature extractor** for wireless channels. Building on the foundation of LWM-v1.0, this enhanced version incorporates key advancements to handle **diverse channel configurations**, improve **generalization**, and process **larger, more complex datasets**. As a state-of-the-art foundation model, LWM-v1.1 leverages transformers to extract refined representations from simulated datasets like DeepMIMO and real-world wireless data.

### **How is LWM-v1.1 built?**

The LWM-v1.1 architecture is built on transformers, designed to capture **fine-grained and global dependencies** in wireless channel data. The model employs an updated version of **Masked Channel Modeling (MCM)**, increasing the masking ratio to make pretraining more challenging and effective. With **2D patch segmentation**, the model learns intricate relationships across both antennas and subcarriers, while **bucket-based batching** ensures efficient processing of variable-sized inputs. These enhancements make LWM-v1.1 highly scalable and adaptable, offering robust embeddings for diverse scenarios.

### **What does LWM-v1.1 offer?**

LWM-v1.1 provides a versatile feature extraction framework for wireless communication and sensing tasks. Pretrained on a larger and more diverse dataset, it generalizes well across environments—from dense urban cities to synthetic setups—capturing channel characteristics that facilitate reliable performance. With increased capacity and optimized pretraining, LWM-v1.1 embeddings are even more refined, enabling improved results across downstream applications.

### **How is LWM-v1.1 used?**

LWM-v1.1 is designed to be seamlessly integrated into downstream tasks as a source of high-quality **embeddings**. By feeding raw wireless channel data into the model, users obtain contextualized representations that capture critical spatial relationships and dependencies. These embeddings enable efficient and accurate performance in applications such as **beam prediction**, **LoS/NLoS classification**, and **channel estimation**—even with limited labeled data.

### **Advantages of Using LWM-v1.1**

- **Enhanced Flexibility**: Handles diverse channel configurations with no size limitations.
- **Refined Embeddings**: Improved feature extraction through advanced pretraining and increased model capacity.
- **Efficient Processing**: Memory-optimized with bucket-based batching for variable-sized inputs.
- **Broad Generalization**: Trained on a larger, more diverse dataset for reliable performance across environments.
- **Task Adaptability**: Fine-tuning options enable seamless integration into a wide range of applications.

---

## **Overview of Main Changes in LWM-v1.1**
1. **No channel size limitation**  
2. **Larger and more diverse pretraining dataset**  
3. **Fine-tuning capabilities for task-specific embedding generation**  
4. **Increased model capacity**  
5. **2D patch segmentation for realistic learning**  
6. **Challenging MCM task with higher masking ratio**  
7. **Support for larger input sizes**  
8. **Optimized training strategy**  
9. **Improved computational efficiency**  

---

## **Detailed Explanation of Changes in LWM-v1.1**

### **No Channel Size Limitation**  
In **LWM-v1.0**, the model was pre-trained on a single (N, SC) = (32, 32) pair, which limited its generalization to other channel configurations. Wireless communication systems in the real world exhibit vast variability in the number of antennas (N) at base stations and subcarriers (SC). To address this limitation, **LWM-v1.1** was pre-trained on **20 distinct (N, SC) pairs**, ranging from smaller setups like (8, 32) to more complex setups like (128, 64). This variety enables the model to effectively handle diverse channel configurations and ensures robust generalization without overfitting to specific configurations.

To handle variable-sized inputs efficiently, we implemented **bucket-based batching**, where inputs of similar sizes are grouped together. For example, channels with sizes (32, 64) and (16, 128) are placed in the same bucket, avoiding the excessive padding common in traditional batching approaches. This not only saves memory but also ensures computational efficiency during training. Furthermore, validation samples were drawn as **20% of each bucket**, maintaining a balanced evaluation process across all input sizes.

This approach eliminates the rigidity of fixed channel sizes and positions LWM-v1.1 as a versatile model capable of adapting to real-world wireless systems with varying configurations.

---

### **Larger and More Diverse Pretraining Dataset**  
Generalization is a critical aspect of any foundation model. In **LWM-v1.1**, we significantly expanded the training dataset to cover more diverse scenarios and environments. We added **seven new city scenarios**—Charlotte, Denver, Oklahoma, Indianapolis, Fort Worth, Santa Clara, and San Diego—to enrich the model’s exposure to a variety of urban layouts. To enhance the spatial resolution of the training data, we reduced the grid spacing between user locations in the DeepMIMO city scenarios from **2.5m to 1m**, resulting in a higher density of user positions. This adjustment required re-performing ray tracing for all scenarios to generate high-resolution wireless channel data.

Additionally, we introduced **channels from multiple base stations** in each scenario, with distinct (N, SC) pairs to ensure the model encounters a broad range of channel characteristics. This diversity mirrors the variability found in real-world deployments, such as urban, suburban, and rural environments. By exposing LWM-v1.1 to this diversity, the model gains the ability to generalize across environments with distinct propagation characteristics, making it more reliable and versatile.

---

### **Fine-Tuning for Task-Specific Embedding Generation**  
While pretraining provides a robust feature extractor, downstream tasks often require tailored embeddings. In **LWM-v1.1**, we introduced **fine-tuning options** that give users the flexibility to customize the model for specific tasks. Users can now **freeze specific layers** of the model, allowing the remaining layers to adapt to task-specific requirements. This feature is particularly valuable for tasks prone to overfitting, such as **LoS/NLoS classification**, where excessive training on all layers can lead to suboptimal generalization.

To further streamline task-specific adaptation, we provided **default classification and regression heads** for downstream tasks. Users can also define their own custom heads to suit unique requirements, ensuring maximum flexibility and adaptability.

---

### **Increased Model Capacity**  
LWM-v1.1 significantly enhances the model's ability to extract complex features by increasing the **embedding size from 64 to 128**. This increase more than quadruples the model's parameter count, raising it from **600K to 2.5M**. The larger embedding size allows the model to represent more intricate relationships within channel data, improving its performance on challenging tasks such as **beam prediction** and **channel estimation**.

This change directly impacts the quality of the embeddings, making them more expressive and robust across a variety of downstream tasks, even in scenarios with limited labeled data.

---

### **Challenging MCM Task with Higher Masking Ratio**  
The **Masked Channel Modeling (MCM)** task lies at the core of LWM’s pretraining methodology. In **LWM-v1.1**, we made the task more challenging by increasing the **masking ratio from 15% to 40%**. This means that a larger portion of the channel data is masked during training, requiring the model to infer the missing information from contextual dependencies.

This enhancement forces the model to rely on deeper spatial relationships between antennas and subcarriers, rather than learning superficial patterns. As a result, LWM-v1.1 produces embeddings that are more robust and better equipped to handle real-world scenarios with incomplete or noisy data.

---

### **Support for Larger Input Sizes**  
Wireless communication systems are increasingly handling larger channels with higher dimensions. To accommodate these demands, we increased the **maximum sequence length** from **128 to 512** in **LWM-v1.1**. This change enables the model to process larger and more detailed channel data without modification, broadening its applicability to high-dimensional wireless tasks. This ensures that LWM-v1.1 remains relevant as the scale and complexity of wireless systems continue to grow.

---

### **2D Patch Segmentation for Realistic Learning**  
In **LWM-v1.0**, patches were segmented based on a single dimension, typically grouping elements from different subcarriers within the same antenna. In **LWM-v1.1**, we introduced **2D patch segmentation**, where patches now combine elements from both antennas and subcarriers. This reflects real-world wireless channel dependencies more accurately, as the relationship between antennas and subcarriers is critical in practical deployments.

This multidimensional segmentation increases the complexity of the MCM task, requiring the model to learn deeper and more meaningful dependencies within the data. By better aligning the training methodology with real-world conditions, LWM-v1.1 further enhances its ability to generalize and perform in practical scenarios.

---

### **Optimized Training Strategy**  
Training large models requires carefully designed optimization techniques to ensure smooth convergence and generalization. In **LWM-v1.1**, we adopted the **AdamW optimizer**, which improves weight regularization and prevents overfitting compared to traditional Adam. The learning rate schedule was also refined, incorporating an **85-step warmup phase** followed by **cosine decay**. This strategy ensures that the model transitions smoothly from the initial training phase to convergence, maintaining stability and improving overall performance.

---

### **Improved Computational Efficiency**  
To balance computational efficiency with performance, we reduced the number of **attention heads per layer from 12 to 8** in **LWM-v1.1**. This reduction decreases the computational load during both training and inference, making the model more efficient without significantly affecting its ability to extract meaningful features. The streamlined architecture ensures that LWM-v1.1 is not only powerful but also practical for deployment in resource-constrained environments.

---

### **Why These Changes Were Necessary**  
The updates in LWM-v1.1 were driven by real-world demands for greater flexibility, scalability, and performance in wireless communication tasks. Removing channel size limitations and diversifying the dataset address the variability inherent in wireless environments. Increasing model capacity and enhancing the MCM task improve the quality of embeddings, while optimized training strategies and computational efficiency make the model practical for a wide range of applications. These changes make LWM-v1.1 a significant step forward, ensuring its relevance and impact in advancing wireless communication research.

---

## **Conclusion**  
**LWM-v1.1** represents a major leap forward in wireless communication modeling, offering robust scalability, increased generalization, and adaptability to a wide variety of tasks. From enriched training datasets and challenging pretraining objectives to enhanced model capacity and efficient input handling, LWM-v1.1 provides a powerful foundation for wireless communication research and applications.  

---

### **Try It Now!**  
Explore **LWM-v1.1** on Hugging Face with preloaded datasets, fine-tuning options, and pretrained models to kickstart your projects.  
[👉 Access the model here!](https://huggingface.co/wi-lab/lwm-v1.1)

---

Please cite the following paper if you use the LWM model or any modifiled parts:
```
@misc{alikhani2024largewirelessmodellwm,
      title={Large Wireless Model (LWM): A Foundation Model for Wireless Channels}, 
      author={Sadjad Alikhani and Gouranga Charan and Ahmed Alkhateeb},
      year={2024},
      eprint={2411.08872},
      archivePrefix={arXiv},
      primaryClass={cs.IT},
      url={https://arxiv.org/abs/2411.08872}, 
}
```
