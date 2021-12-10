# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

For this project, I chose the Resnet18 model that could be fine-tuned to predict classes on the dogbreed classification dataset. I primarily chose this dataset as I do not have access to a broad selection of datasets nor do I have preference towards another dataset at this time - plus, dogs are fascinating to me. 

The two hyperparameters that I chose to tune are batch size and learning rate. Originally I had included epochs as well, which might be good if there are many evaluations possible, but for the 4 total jobs I ran I thought that including epochs might lead to sub-optimal results (as typically, training results get better with more epochs, and when using a small number of epochs as we have time and resources for, that trend generally holds well).

For batch size, I used a categorical approach of 8 & 32, similar to examples given in the course. For learning rate, however, rather than a continous spectrum I chose three values and made this categorical as well - 0.001, 0.003, and 0.01 - this was based on my previous experience in learning ML (modifying the learning rate by a factor of 3) as well as previous experience in optimization (reducing the design space to get meaningful results). I recognize this won't provide me a global optimum of settings, but should be effective at identifying the best settings with available time and resources.

Here is a screenshot of the completed training jobs, including objective metric values:

![Completed Tuning Job](https://irrer.net/aws_nd/hpo_tuning_completed.png)

Here is a screenshot of the code that retrieves the best hyperparameter values of 0.003 learning rate and 32 batch size:

![Best Tuning Hyperparameters](https://irrer.net/aws_nd/hpo_tuning_parameters.png)

More information can be found in the notebook, as detailed there.

## Debugging and Profiling
To perform model debugging and profiling, I added the necessary hooks in my train_model.py file that captured statistics about the running training job. In my notebook, I added rules as well as debugger hook config and profiler config that would capture this data and store it in my s3 bucket. 

### Results
My biggest takeaway from debugging the model was looking at the consistent, but low, GPU utilization and memory utilization. This reflects general experience I've observed online - that machine learning models aren't always running optimally on GPUs, due to their varied architecture and variances in the underlying ML problem/models. This is why, I believe, tuning the batch size for a specific problem can lead to significant gains sometimes - enabling fuller use of the GPU resources, but being cautious to not exceed available memory. Here is the GPU/memory utilization for my training job:

![Debugging Metrics](https://irrer.net/aws_nd/debugging.png)

My observations on profiling are more limited; none of the rules were triggered, so there was no observation that helped to improve my job, except that confirming it was running correctly. The profiler PDF is attached in the submission for review, as well as shown in the notebook html.

## Model Deployment
After trying multiple times to deploy my endpoint programatically from the training job or tuning job, which kept failing due to debugging issues with the entrypoint script, I took a manual approach. I modified the entry point Python script manually, and packaged it in the code directory of a new model.tar.gz file. I then manually defined a Model in Sagemaker, using a previously trained model as a reference. Then, created an endpoint configuration and finally deployed that endpoint. Notably, I was having trouble initially when deploying an endpoint to an ml.g4dn.xlarge instance - trying to use the GPU for inference - but eventually was successful deploying to an ml.m4.xlarge instance and modifying the inference code to run on a CPU instead. Here is my deployed endpoint, in service:

![Deployed Endpoint](https://irrer.net/aws_nd/endpoint.png)

To get meaningful data into and out of the endpoint, it must be referenced using numpy serializer and deserializers:

![Endpoint Serializer and Deserializer](https://irrer.net/aws_nd/endpointserializers.png)

The endpoint is referenced and queried in my notebook. Here is an example image used for querying:

![Golden Retriever](https://irrer.net/aws_nd/goldenretriever.png)

First, an image has to be loaded and transformed into a 224x224 RBG image via a torchvision transform, then converted to a numpy array, and fed into the endpoint. The resulting numpy array can then be checked for max value (ie, class) - the predicted value ranges from 0-132, while the labels (as part of the directory name) range from 1-133, so I offset the predicted value by 1. The predicted results can also be plotted. Here is a function I defined to handle all of this, given the path of an input image:

![Inference Function](https://irrer.net/aws_nd/dogbreedclassification.png)



## Standout Suggestions
While querying the images, I also generate a plot that shows all responses to look for interesting classes which are high in value. For the golden retriever, I was curious about the 2nd highest predicted class (label 128+1=129), and queried that class further. At first glance, it looks completely different (length of fur, fur color), but upon examining the facial features more closely, it does resemble a golden retriever's facial features:

![Tibetan Mastiff](https://irrer.net/aws_nd/tibetan_mastiff.png)


