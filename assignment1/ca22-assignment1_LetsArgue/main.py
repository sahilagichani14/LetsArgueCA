import csv, json, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def main():
    #Path of the csv file provided 
    csv_path=input("give path for csv file:")
    with open(csv_path,'r', encoding='ISO-8859-1') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter='\t')
        reader_list = []
        for line in csv_reader:
            reader_list.append(dict(line))

    res = []

    #extracting all required quality dimentions from the csv file
    for row in reader_list:
        res.append(dict((k, row[k]) for k in ['#id','issue','stance','argumentative','overall quality','effectiveness','argument','emotional appeal'] if k in row))
            
    final_data = []

    def str_to_float(temp_str):
        if temp_str=='':
            return (0.0)
        else:
            return (float(temp_str))

    #making a list of all three annotators per argument 
    for i in range(len(res)):    
        if i%3==0:
            temp_argumentative = []
            temp_argumentative.append(res[i]['argumentative'])
            temp_argumentative.append(res[i+1]['argumentative'])
            temp_argumentative.append(res[i+2]['argumentative'])
            
            temp_overall_quality = []
            temp_overall_quality.append(str_to_float(res[i]['overall quality'][:1]))
            temp_overall_quality.append(str_to_float(res[i+1]['overall quality'][:1]))
            temp_overall_quality.append(str_to_float(res[i+2]['overall quality'][:1]))
            
            temp_effectiveness = []
            temp_effectiveness.append(str_to_float(res[i]['effectiveness'][:1]))
            temp_effectiveness.append(str_to_float(res[i+1]['effectiveness'][:1]))
            temp_effectiveness.append(str_to_float(res[i+2]['effectiveness'][:1]))
            
            temp_emotional_appeal = []
            temp_emotional_appeal.append(str_to_float(res[i]['emotional appeal'][:1]))
            temp_emotional_appeal.append(str_to_float(res[i+1]['emotional appeal'][:1]))
            temp_emotional_appeal.append(str_to_float(res[i+2]['emotional appeal'][:1]))
            
            res[i]['argumentative']=temp_argumentative
            res[i]['overall quality']=temp_overall_quality
            res[i]['effectiveness']=temp_effectiveness
            res[i]['emotional appeal']=temp_emotional_appeal

            if temp_argumentative==['y','y','y']:
                final_data.append(res[i])

    #changing the name of keys as desired
    for i in range(len(final_data)):
        final_data[i]['id'] = final_data[i].pop('#id')
        final_data[i]['issue'] = final_data[i].pop('issue')
        final_data[i]['stance_on_topic'] = final_data[i].pop('stance')
        final_data[i]['argumentative'] = final_data[i].pop('argumentative')
        final_data[i]['argument_quality_scores'] = final_data[i].pop('overall quality')
        final_data[i]['effectiveness_scores'] = final_data[i].pop('effectiveness')
        final_data[i]['emotional_appeal_scores'] = final_data[i].pop('emotional appeal')
        final_data[i]['text'] = final_data[i].pop('argument')

    #dump final data (304 arguments) into json file
    final_wo_emotionalappeal = copy.deepcopy(final_data)

    for element in final_wo_emotionalappeal:
        element.pop('emotional_appeal_scores')
        
    with open('final_output.json', 'w', encoding='ISO-8859-1') as f:
        json.dump(final_wo_emotionalappeal, f, ensure_ascii=False, indent=2)
    
    #Splitting of Data & dumping into train, test, validation   
    with open('final_output.json','r') as f:
        lines = json.load(f)

    train, test = train_test_split(lines, test_size=0.3)
    val, test = train_test_split(test, test_size=0.66)

    with open('train.json', 'w', encoding='ISO-8859-1') as f:
        json.dump(train, f, ensure_ascii=False, indent=2)

    with open('test.json', 'w', encoding='ISO-8859-1') as f:
        json.dump(test, f, ensure_ascii=False, indent=2)

    with open('val.json', 'w', encoding='ISO-8859-1') as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    #Statistics
    plt.style.use('ggplot')

    # converting final data into a Data frame
    df = pd.DataFrame(final_data)

    # calculating average quality score and average effectiveness score per issue
    for i in range(len(df)) :
        df.loc[i,'avg_argument_quality_scores']= (df.loc[i, "argument_quality_scores"][0] + df.loc[i, "argument_quality_scores"][1] + df.loc[i, "argument_quality_scores"][2])/3
        df.loc[i,'avg_effectiveness_scores']= (df.loc[i, "effectiveness_scores"][0] + df.loc[i, "effectiveness_scores"][1] + df.loc[i, "effectiveness_scores"][2])/3
        df.loc[i,'avg_emotional_appeal_scores']= (df.loc[i, "emotional_appeal_scores"][0] + df.loc[i, "emotional_appeal_scores"][1] + df.loc[i, "emotional_appeal_scores"][2])/3

    # plotting Avg Quality Score per Issue
    df_issues_effectivescore = df[['issue', 'avg_argument_quality_scores']]  
    df_grpbyissue_quality = df_issues_effectivescore.groupby(["issue"]).mean()
    df_grpbyissue_quality.plot(figsize=(8, 4), kind='bar',title='Avg Quality Score per Issue',grid=True)
    plt.xlabel('Issues')
    plt.ylabel('Average Quality Score')
    plt.xticks(rotation = 90)
    plt.show()

    # plotting Avg Effectiveness per Issue
    df_issues_qualityscores = df[['issue', 'avg_effectiveness_scores']]  
    df_grpbyissue_effectiveness = df_issues_qualityscores.groupby(["issue"]).mean()
    df_grpbyissue_effectiveness.plot(figsize=(8, 4), kind='bar', color='g',title='Avg Effectiveness per Issue',grid=True)
    plt.xlabel('Issues')
    plt.ylabel('Average Effectiveness Score')
    plt.xticks(rotation = 90)
    plt.show()

    # Argumentative Quality VS Emotional Appeal: How emotioanal appeal effects an argument
    scores = ['avg_argument_quality_scores','avg_emotional_appeal_scores']
    dataset = df.groupby('issue')[scores].mean()
    indx = np.arange(len(scores))
    score_label = np.arange(0,150000 , 500)
    topics = df.issue.unique()
    argumentative_quality_mean = list(dataset['avg_argument_quality_scores'])
    emotional_appeal_mean = list(dataset['avg_emotional_appeal_scores'])
    x_axis = np.arange(len(topics))
    # Multi bar Chart
    plt.bar(x_axis -0.2, argumentative_quality_mean, width=0.4, label = 'Argumentative Quality Mean', align='center')
    plt.bar(x_axis +0.2, emotional_appeal_mean, width=0.4, label = 'Emotional Appeal Mean', align='center')
    plt.plot(kind ='bar', figsize=(15, 10), title='Argumentative Quality VS Emotional Appeal',grid=True)
    plt.autoscale(tight=True)
    plt.xticks(x_axis, topics, rotation=90)
    plt.legend()
    plt.show()

    print("it works!")
    pass

if __name__ == '__main__':
    main()