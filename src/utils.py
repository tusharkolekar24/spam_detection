import pickle
import os
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,f1_score
from sklearn.metrics import precision_score,classification_report
from sklearn.metrics import roc_curve,roc_auc_score

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

def get_area_under_curve(true,pred,file_path,model_name,types):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # roc curve for tpr = fpr 
    random_probs = [0 for i in range(len(true))]
    p_fpr, p_tpr, _ = roc_curve(true, random_probs, pos_label=1)

    # roc curve for models
    fpr1, tpr1, thresh1 = roc_curve(true, pred, pos_label=1)

    auc_score1 = roc_auc_score(true,pred)
    
    fig = plt.figure()
    plt.style.use('seaborn')
    plt.plot(fpr1, tpr1, linestyle='--',color='orange', label=model_name)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    
    # title
    plt.title('ROC curve')
    
    # x label
    plt.xlabel('False Positive Rate')
    
    # y label
    plt.ylabel('True Positive rate')
    
    plt.legend(title=f"AUC Score:{np.round(auc_score1,6)}")
    
    #
    plt.savefig(os.path.join(file_path,f'{model_name}_{types}_ROC.jpg')) #,dpi=300
    
    plt.close(fig)

def Evaluate_Models(X_train,X_test,y_train,y_test):
      import matplotlib.pyplot as plt
      auc_fig_train = plt.figure()
      auc_fig_test  = plt.figure()
      pickle_file_path = os.path.join(os.getcwd(),'artifacts')
      report = {'model_name':[],'train_accuracy':[],'test_accuracy':[],'precision_score_train':[],'precision_score_test':[],
                'recall_score_train':[],'recall_score_test':[],'f1_score_train':[],'f1_score_test':[],
                'auc_score_train':[],'auc_score_test':[]
               }
      
      for pkl_files in os.listdir(pickle_file_path):
          if pkl_files.endswith('.pkl'):
                print(pkl_files)
                read_model_store_pkl = pickle.load(open(os.path.join(pickle_file_path,pkl_files),'rb'))
                y_pred_train = read_model_store_pkl.predict(X_train)
                y_pred_test  = read_model_store_pkl.predict(X_test)
                y_pred_train_proba = read_model_store_pkl.predict_proba(X_train)
                y_pred_test_proba  = read_model_store_pkl.predict_proba(X_test)

                if not os.path.exists(os.path.join(os.getcwd(),'artifacts','Confusion_Matrix')):
                     os.mkdir(os.path.join(os.getcwd(),'artifacts','Confusion_Matrix'))

                if not os.path.exists(os.path.join(os.getcwd(),'artifacts','AUC')):
                     os.mkdir(os.path.join(os.getcwd(),'artifacts','AUC'))

                file_path = os.path.join(os.getcwd(),'artifacts','Confusion_Matrix')
                file_path_auc = os.path.join(os.getcwd(),'artifacts','AUC')

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
                
                get_area_under_curve(true       = y_train,
                                     pred       = y_pred_train,
                                     file_path  = file_path_auc,
                                     model_name = pkl_files.replace(".pkl",""),
                                     types      = 'train')
                
                get_area_under_curve(true       = y_test,
                                     pred       = y_pred_test,
                                     file_path  = file_path_auc,
                                     model_name = pkl_files.replace(".pkl",""),
                                     types      = 'test')
                
                report['model_name'].append(pkl_files.replace(".pkl",""))
                report['train_accuracy'].append(accuracy_score(y_train,y_pred_train))
                report['test_accuracy'].append(accuracy_score(y_test,y_pred_test))
                report['precision_score_train'].append(precision_score(y_train,y_pred_train))
                report['precision_score_test'].append(precision_score(y_test,y_pred_test))
                report['recall_score_train'].append(recall_score(y_train,y_pred_train))
                report['recall_score_test'].append(recall_score(y_test,y_pred_test))
                report['f1_score_train'].append(f1_score(y_train,y_pred_train))
                report['f1_score_test'].append(f1_score(y_test,y_pred_test)) #np.argmax(pred_prob1,axis=1)
                report['auc_score_train'].append(roc_auc_score(y_train,np.argmax(y_pred_train_proba,axis=1)))
                report['auc_score_test'].append(roc_auc_score(y_test,np.argmax(y_pred_test_proba,axis=1)))
                # report['confusion_matrix_train'].append(confusion_matrix(y_train,y_pred_train))
                # report['confusion_matrix_test'].append(confusion_matrix(y_test,y_pred_test))


      return report               