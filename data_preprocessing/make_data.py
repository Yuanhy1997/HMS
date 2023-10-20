import os
import json
import random
import time
import re
import pickle
import traceback
from tqdm import tqdm
import pandas as pd


## Functions to process the data
def DataClean(data,labels_path,data_path=None,delete=None):

    print('\n\nImporting and filtering database...')

    if data is not None and not data.empty:
        notes = data
    else:
        notes = pd.read_csv(data_path)

    ints_str = '0123456789-#[]' # characters that usually main categories don't start with

    print('\n\nSplitting each note into sections:\n\n')

    notes_sections = {}

    for note_index in tqdm(range(notes.shape[0])):
        note = notes['text'][note_index].replace('\n\n\n\n','\n').replace('\n\n\n','\n').replace('     ','\n')
        paragraphs = note.split('\n')

        subsections, new_section = [], ' '
        for p in paragraphs:
            line = p.strip()
            if len(line)>0 and ':' in line and not (line[line.find(':')-1] in ints_str) and not(line[0] in ints_str):
                subsections.append([new_section.strip()])
                new_section = p + ' '
            else:
                new_section += p + ' '
        subsections.append([new_section])
        subsections.pop(0)

        note_sect_tit,note_sect_par = [],[]
        for sect in subsections:
            note_sect_tit += [str(*sect)[0:str(*sect).find(':')]]
            note_sect_par += [str(*sect)[str(*sect).find(':')+1:].strip()]
        note_df = pd.DataFrame({'title':note_sect_tit,'category':'','text':note_sect_par, 'label':''})
        notes_sections[notes['note_id'][note_index]] = note_df

    f = open(labels_path, 'r')
    obj_label = f.readlines()
    obj_label_dict = {}
    i = 0
    for s in obj_label:
        i += 1
        if '/' in s:
            buffer = s.strip('\n').lower().split('/')
            for item in buffer:
                obj_label_dict[item] = i
        else:
            obj_label_dict[s.strip('\n').lower()] = i
    f.close()

    for key in tqdm(list(notes_sections.keys())):
        buffer = 'begin_title'
        t = list(notes_sections[key]['title'])
        for idx in range(len(t)):
            for item in list(obj_label_dict.keys()):
                if item in t[idx].lower() and len(t[idx].lower())>2:
                    buffer = item
                    notes_sections[key]['category'][idx] = buffer
                    notes_sections[key]['label'][idx] = obj_label_dict[buffer]
                    break
            notes_sections[key]['category'][idx] = buffer
            notes_sections[key]['label'][idx] = obj_label_dict[buffer]

    notes_sections_output = {}
    row_id  = notes_sections.keys()
    for key in tqdm(row_id):
        buffer = ''
        note_sect_tit, note_sect_par, note_sect_lab = [], [], []
        for i in range(len(notes_sections[key]['category'])):
            if buffer != notes_sections[key]['category'][i]:
                buffer = notes_sections[key]['category'][i]
                note_sect_tit.append(buffer)
                note_sect_lab.append(notes_sections[key]['title'][i])
                note_sect_par.append(notes_sections[key]['text'][i])
                # if buffer == 'followup instruction' or buffer == 'follow up' or buffer == 'follow-up':
                #     break
            else:
                note_sect_par[-1] = note_sect_par[-1] + ' ' + notes_sections[key]['title'][i] + ' ' + notes_sections[key]['text'][i]
        note_df = pd.DataFrame({'title': note_sect_tit, 'text': note_sect_par, 'label': note_sect_lab})
        notes_sections_output[key] = note_df


    notes_sections = notes_sections_output

    if delete != None:
        for key,value in notes_sections.items():
            notes_sections[key] = notes_sections[key][~notes_sections[key]['label'].isin(delete)]

    return notes_sections



df  = pd.read_csv("/content/malignant_neoplasm_updated_first_100_rows.csv").iloc[:10,]


NotesSections = DataClean(data=df,
                          labels_path="/content/labels.txt",
                          delete=['Name','Admission Date','Discharge Date','Date of Birth','Followup Instructions'])

Data = {}

for key, value in NotesSections.items():

    Data[key] = {}

    Strings = NotesSections[key]['text'].tolist()

    ## Delete some useless infomation
    paragraphs = [s for s in Strings if s != ""]

    collection = []

    for para in paragraphs:

        Sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<! [a-z]\.)(?<![A-Z][a-z][a-z]\.)(?<=\.|\?|\!)\"*\s*\s*(?:\W*)(?<![A-Z])', para)

        collection = collection + [s for s in Sentences if len(s) > 10]