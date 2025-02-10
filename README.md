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

Then I wrote a function to start with a base image and use cross entropy loss calculations to make 

# Classification Model

