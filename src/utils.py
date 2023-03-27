import pickle
import os
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,f1_score
from sklearn.metrics import precision_score,classification_report

def Save_Object(model,model_name):
        # print(model_name)
        with open (os.path.join(os.getcwd(),'artifacts',f'{model_name.replace(" ","_")}.pkl'),'wb') as files:
            pickle.dump(model,files)    
            
def get_confusion_matrix(true,pred,file_path,model_name,types):
    import matplotlib.pyplot as plt
    import seaborn as sns

    #plt.style.use('seaborn')
    fig = plt.figure(figsize=(4,3))  
    cm = confusion_matrix(true,pred)
    sns.heatmap(cm,annot=True,cmap="Blues",
                fmt='g',xticklabels=['Spam','Ham'],
                yticklabels=['spam','ham'])
    plt.title("Confusion Matrics Statistics")
    
    fig.savefig(os.path.join(file_path,f"{model_name}_{types}.jpg"))

def Evaluate_Models(X_train,X_test,y_train,y_test):
      pickle_file_path = os.path.join(os.getcwd(),'artifacts')
      report = {'model_name':[],'train_accuracy':[],'test_accuracy':[],'precision_score_train':[],'precision_score_test':[],
                'recall_score_train':[],'recall_score_test':[],'f1_score_train':[],'f1_score_test':[],
                #'confusion_matrix_train':[],'confusion_matrix_test':[]
               }
      
      for pkl_files in os.listdir(pickle_file_path):
          if pkl_files.endswith('.pkl'):
                print(pkl_files)
                read_model_store_pkl = pickle.load(open(os.path.join(pickle_file_path,pkl_files),'rb'))
                y_pred_train = read_model_store_pkl.predict(X_train)
                y_pred_test  = read_model_store_pkl.predict(X_test)
                
                if not os.path.exists(os.path.join(os.getcwd(),'artifacts','Confusion_Matrix')):
                     os.mkdir(os.path.join(os.getcwd(),'artifacts','Confusion_Matrix'))

                file_path = os.path.join(os.getcwd(),'artifacts','Confusion_Matrix')

                get_confusion_matrix(true       = y_train,
                                     pred       = y_pred_train,
                                     file_path  = file_path,
                                     model_name = pkl_files.replace(".pkl",""),
                                     types      = 'train')

                get_confusion_matrix(true      = y_test,
                                     pred      = y_pred_test,
                                     file_path = file_path,
                                     model_name=pkl_files.replace(".pkl",""),
                                     types     ='test')
                
                report['model_name'].append(pkl_files.replace(".pkl",""))
                report['train_accuracy'].append(accuracy_score(y_train,y_pred_train))
                report['test_accuracy'].append(accuracy_score(y_test,y_pred_test))
                report['precision_score_train'].append(precision_score(y_train,y_pred_train))
                report['precision_score_test'].append(precision_score(y_test,y_pred_test))
                report['recall_score_train'].append(recall_score(y_train,y_pred_train))
                report['recall_score_test'].append(recall_score(y_test,y_pred_test))
                report['f1_score_train'].append(f1_score(y_train,y_pred_train))
                report['f1_score_test'].append(f1_score(y_test,y_pred_test))
                
                # report['confusion_matrix_train'].append(confusion_matrix(y_train,y_pred_train))
                # report['confusion_matrix_test'].append(confusion_matrix(y_test,y_pred_test))
      return report               