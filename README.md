# captcha_task

# Word List
As part of the task, I started off by creating a word list that was handleable from: https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt

Then I created a word list -> word_lists/custom_word_list
I narrowed it down to a 1000 word list -> word_lists/custom_word_list_1000
For training models, I further narrowed it down to a 100 word list with number of characters ranging from 4 to 7 -> word_lists/custom_list_100
You can see the code for this in word_list.ipynb

# Making images
I started generating the dataset using Python, but due to the fact that the model was not creating any specific results as I wanted due to the limitation, I switched to dataset creation using Javascript as it enabled me to use the ctx (a turtle based) code to generate models which turned out to be easy. This was also the first time that I have ever used javascript code in any part of a model training exercise.

I have created codes for 
- Easy set: generate_easy.js
- Hard (hollow): generate_hard_hollow.js
- Hard (nonhollow): generate_hard_nonhollow.js
- Hard (hollow and random like actual captcha): generate_hard_hollow_random.js
- Bonus (red): generate_bonus_red.js
- Bonus (green): generate_bonus_green.js

I then organized the images into a folder format - dataset/`<word>`/`<captcha_images with the word>`. Code can be seen in organize_for_classification.ipynb

Dataset - https://drive.google.com/file/d/1_rZMkI4wx2eydPWP1R1NOKijH0_R2DGE/view?usp=drive_link
Dataset with bonus - https://drive.google.com/file/d/1hpmfsm86egdmzOZ_2SozBMst4C5YK1RA/view?usp=sharing

# Model reversing
You can see related stuff at /model_reversing

In between i got busy with creating question for The Deccan CTF which the hacking club conducted on 8th-9th. I created a question that I believer has very much importance and helped me learn more about image classification models and so forth (only one person was able to cract the question in the CTF)

The code and the results can be seen in model_reversing/train_and_reverse.ipynb where i purposefully overfit the model into images for 9 classes s1-s9
s1-T
s2-R
s3-Y
s4-H
s5-A
s6-C
s7-K
s8-M
s9-E

Then I wrote a function to start with a base image and use cross entropy loss calculations to get the images back from the model without even knowing what the model was.

In the CTF, participants were asked to find the same and submit the flag as 0x1337{TRYHACKME}

# Classification Model
You can see classification and related stuff at /classifier_model

For the image classification model, i decided to go with a Resnet-50 model pretrained on ImageNet. I chose Resnet, because as part of my limited experience with vision based models Resnet had given me good results - I used Resnet for my megathon task, some project with model quanitzation that I did for SPCRC and Qualcomm over last winter etc. I took resnet-50 over resnet18 because i wanted the more layers and the accuracy. 

I took close to 20 minutes per epoch on colab with T4 gpu so I had to implement the saving the model when accuracy has increased as compared to before.

64 and 128 batch sizes would give me better stability but those took too long to load and/or were bigger that available GPU VRAM on colab and my laptop which prevented me from using them. 

So I switched to a 32 batch size training plan with a hopeful 50 epoch due to the large dataset size

100 images x 481(minimum image size) = 48100 (if all the words had just 4 letters).

I could only train to 10 epochs before I lost my daily colab GPU credits, so I trained for 10 epochs

I used a DataLoader to load the dataset structure and efficiently (randomly) split been train and validation sets as required in each epochs.

```
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])
```
Resize: Resizes images to 224x224, as required by ResNet50.
RandomHorizontalFlip: Randomly flips images horizontally for data augmentation.
RandomRotation: Rotates images by up to 20 degrees for augmentation.
ColorJitter: Randomly changes brightness, contrast, and saturation for augmentation.
ToTensor: Converts images to PyTorch tensors.
Normalize: Normalizes images using ImageNet mean and standard deviation (required for pre-trained models).

```
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```
Cross-Entropy loss as it was suitable for this multi-class classifciationn tasks.
Adam - It adapted the learning rate during training.

```
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # Print training statistics
    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
```

model.train(): Sets the model to training mode.
images.to(device), labels.to(device): Moves data to the selected device.
outputs = model(images): Performs a forward pass.
loss = criterion(outputs, labels): Computes the loss.
optimizer.zero_grad(): Clears old gradients.
loss.backward(): Computes gradients.
optimizer.step(): Updates model weights.
running_loss, correct, total: Track loss and accuracy for each epoch.

```
   model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Track statistics
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # Print validation statistics
    val_loss = val_loss / len(val_loader)
    val_acc = 100.0 * correct / total
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
```
model.eval(): Sets the model to evaluation mode.
with torch.no_grad(): Disables gradient computation for faster inference.
Validation statistics are computed similarly to training statistics.

There are close to 480-54000 images per class.

A test of the classification model can be seen in classifier_model.

# Generation
You can see code and everything at /generator_model


I chose CRNN for this: CRNN (Convolutional Recurrent Neural Network) is a great choice for CAPTCHA text recognition because it combines CNNs for feature extraction and RNNs for sequence modeling, making it ideal for handling variable-length and distorted text in images.
CNN - edges, shapes, textures
RNN - relation between characters
CTC Loss - Sequence Alignment

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # Leave 0 for blank token
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}
- character index

Because of limitations in copmute power, I limited number of images from each dataset to 1 easy(the only generated) and 20 of everything else and 40 of the red. I tried playing with it. Results are not the best, some letters get wrong most of the time.
If i had more compute power, I could try a better version

If you see the resultsm, they are very bad but that is because I had to reduce the number of epochs, batch size and general quality of dataset. Can be made better something Clip which i was not able to implement efficiently for the computing resource i had.

nn.Sequential -> layer pipeline
nn.Conv2D -> grayscale to feature maps with 3x3 kernel
nn.RELU -> applies non-linearity (most important features)
nn.MaxPool2d -> Downsamples
nn.Conv2d -> deeper features from most important fratures
nn.RELU -> again most important features
nn.MaxPool2d -> Downsamples
repeated couple more times

nn.LSTM -> flattening images, RNN model to character sequences (CNN has reduced image width by 16)
nn.Linear -> convert to probabilities for each character

Bidirectional LSTM -> for forward and backward context (red and green)

CTC Loss -> handles unaligned losses used as loss function for the training
Adam optimizer


logits = model(image)
log_probs = logits.log_softmax(2)
ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True) - averages loss over the batch
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths) - model prediction (Txbatchxnum_classes)

Better than crossentropy loss and labels are not fixed. CTC is better than Crossentropy for ocr like tasks similar to captcha

I understand the results are bad but with more time, gpu power and some more knowledge of ml concepts I can make something better

# Paper Reading
paper reading report can be seen in /paper_report