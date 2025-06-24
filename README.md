# onwa_project

this code has three main sections-
1) creating datsets from the alerts folder (refer to final_dataset_creation.py)
2) training model using learning transfer - mobilevnet2 (refer to training_model_learning_transfer.py)
3) finally it needs to run the model and identify alerts through a webcam and log the identified symbols and noting down duration and confidence percentage in the log
   (refer to predication_cam_logger.py)

   SOME PREQUISTES FOR STEP 3,
   IF YOU WANT TO RUN THE PROGRAM IN COLLAB
   NEED TO DOWNLOAD AND OPEN FILES/FOLDER IN THE FILES SECTION TO DIRECTLY USE MODEL(below files need to be downloaded and saved locally or to drive if need to upload
   to collab)
   1) best_model.h5
   2) logs folder(traning and validation)
   3) label_map.txt


   these are the primary needs of to run code 3 directly without retraining the model in collab when runtime is over.



4) finally the resultant file created will be alert_logs_verified.csv - which will store the alert_name, duration , confidence and datetime.
