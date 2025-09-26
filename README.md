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

finally inside ```Audio_challange_machine_learning\code\audio_files\16981327\clarity``` is the command of ```\Audio_challange_machine_learning\code\audio_files\16981327> git clone https://github.com/claritychallenge/clarity.git ``` which means i can now access the baseline models and evalutations with the path path_to_baseline_code = Path(r"Audio_challange_machine_learning\code\audio_files\16981327\clarity\recipes\cad_icassp_2026\baseline")
which gives access to the 2 provided baseline methods and the way they layout ranking audio data. 