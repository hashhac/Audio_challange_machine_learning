# Audio_challange_machine_learning
This is my submission and contribution after extensive research into the ai capabilities and enhancements they can bring to hearing aids. I will now create a comparison and demonstration of the most reliable and effective methods of sound separation. 
# make sure to create a venv 
Go into your vs code and make with terminal or use the inbuilt conda etc then please reade throught the created directories inside of code as of where to put things and please use them so stuff always runs reguardless of system as to fix it later is anoying.

# Next of by installing the requirements 
```pip install -r "requirements.txt"```


# ADDED code and datafiles
I have donwloaded the clarity challange audio dataset and saved it in audiofiles meaning there is now a 
```Audio_challange_machine_learning\code\audio_files\16981327``` I have then run the following commands to open the 2 tar files ```tar -xvzf cadenza_clip1_data.train.v1.0.tar.gz  # For training data
tar -xvzf cadenza_clip1_data.valid.v1.0.tar.gz # For validation data````

This github has been changed to include the full clone of pyclarity version 0.80.0 which is why the files here use functions from that which have been provided for ranking and for hearing aid simulation at this point who knows what else. But it is all here ```"Audio_challange_machine_learning\code\audio_files\16981327\clarity"``` where the extracted data from the cadanza data goes ```"Audio_challange_machine_learning\code\audio_files\16981327\cadenza_data"```