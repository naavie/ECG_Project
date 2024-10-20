import pandas as pd
import numpy as np
from glob import glob
import wfdb
import os

DECODE_DICT = dict()
DECODE_DICT["Dx_270492004"] = ("IAVB", "1st degree av block")
DECODE_DICT["Dx_195042002"] = ("IIAVB", "2nd degree av block")
DECODE_DICT["Dx_164951009"] = ("abQRS", "abnormal QRS")
DECODE_DICT["Dx_426664006"] = ("AJR", "accelerated junctional rhythm")
DECODE_DICT["Dx_57054005"] = ("AMI", "acute myocardial infarction")
DECODE_DICT["Dx_413444003"] = ("AMIs", "acute myocardial ischemia")
DECODE_DICT["Dx_426434006"] = ("AnMIs", "anterior ischemia")
DECODE_DICT["Dx_54329005"] = ("AnMI", "anterior myocardial infarction")
DECODE_DICT["Dx_251173003"] = ("AB", "atrial bigeminy")
DECODE_DICT["Dx_164889003"] = ("AF", "atrial fibrillation")
DECODE_DICT["Dx_195080001"] = ("AFAFL", "atrial fibrillation and flutter")
DECODE_DICT["Dx_164890007"] = ("AFL", "atrial flutter")
DECODE_DICT["Dx_195126007"] = ("AH", "atrial hypertrophy")
DECODE_DICT["Dx_251268003"] = ("AP", "atrial pacing pattern")
DECODE_DICT["Dx_713422000"] = ("ATach", "atrial tachycardia")
DECODE_DICT["Dx_29320008"] = ("AVJR", "atrioventricular junctional rhythm")
DECODE_DICT["Dx_233917008"] = ("AVB", "av block")
DECODE_DICT["Dx_251170000"] = ("BPAC", "blocked premature atrial contraction")
DECODE_DICT["Dx_74615001"] = ("BTS", "brady tachy syndrome")
DECODE_DICT["Dx_426627000"] = ("Brady", "bradycardia")
DECODE_DICT["Dx_6374002"] = ("BBB", "bundle branch block")
DECODE_DICT["Dx_698247007"] = ("CD", "cardiac dysrhythmia")
DECODE_DICT["Dx_426749004"] = ("CAF", "chronic atrial fibrillation")
DECODE_DICT["Dx_413844008"] = ("CMI", "chronic myocardial ischemia")
DECODE_DICT["Dx_27885002"] = ("CHB", "complete heart block")
DECODE_DICT["Dx_713427006"] = ("CRBBB", "complete right bundle branch block")
DECODE_DICT["Dx_204384007"] = ("CIAHB", "congenital incomplete atrioventricular heart block")
DECODE_DICT["Dx_53741008"] = ("CHD", "coronary heart disease")
DECODE_DICT["Dx_77867006"] = ("SQT", "decreased qt interval")
DECODE_DICT["Dx_82226007"] = ("DIB", "diffuse intraventricular block")
DECODE_DICT["Dx_428417006"] = ("ERe", "early repolarization")
DECODE_DICT["Dx_13640000"] = ("FB", "fusion beats")
DECODE_DICT["Dx_84114007"] = ("HF", "heart failure")
DECODE_DICT["Dx_368009"] = ("HVD", "heart valve disorder")
DECODE_DICT["Dx_251259000"] = ("HTV", "high t-voltage")
DECODE_DICT["Dx_49260003"] = ("IR", "idioventricular rhythm")
DECODE_DICT["Dx_251120003"] = ("ILBBB", "incomplete left bundle branch block")
DECODE_DICT["Dx_713426002"] = ("IRBBB", "incomplete right bundle branch block")
DECODE_DICT["Dx_251200008"] = ("ICA", "indeterminate cardiac axis")
DECODE_DICT["Dx_425419005"] = ("IIs", "inferior ischaemia")
DECODE_DICT["Dx_704997005"] = ("ISTD", "inferior ST segment depression")
DECODE_DICT["Dx_426995002"] = ("JE", "junctional escape")
DECODE_DICT["Dx_251164006"] = ("c", "junctional premature complex")
DECODE_DICT["Dx_426648003"] = ("JTach", "junctional tachycardia")
DECODE_DICT["Dx_425623009"] = ("LIs", "lateral ischaemia")
DECODE_DICT["Dx_445118002"] = ("LAnFB", "left anterior fascicular block")
DECODE_DICT["Dx_253352002"] = ("LAA", "left atrial abnormality")
DECODE_DICT["Dx_67741000119109"] = ("LAE", "left atrial enlargement")
DECODE_DICT["Dx_446813000"] = ("LAH", "left atrial hypertrophy")
DECODE_DICT["Dx_39732003"] = ("LAD", "left axis deviation")
DECODE_DICT["Dx_164909002"] = ("LBBB", "left bundle branch block")
DECODE_DICT["Dx_445211001"] = ("LPFB", "left posterior fascicular block")
DECODE_DICT["Dx_164873001"] = ("LVH", "left ventricular hypertrophy")
DECODE_DICT["Dx_370365005"] = ("LVS", "left ventricular strain")
DECODE_DICT["Dx_251146004"] = ("LQRSV", "low qrs voltages")
DECODE_DICT["Dx_54016002"] = ("MoI", "mobitz type i wenckebach atrioventricular block")
DECODE_DICT["Dx_164865005"] = ("MI", "myocardial infarction")
DECODE_DICT["Dx_164861001"] = ("MIs", "myocardial ischemia")
DECODE_DICT["Dx_698252002"] = ("NSIVCB", "nonspecific intraventricular conduction disorder")
DECODE_DICT["Dx_428750005"] = ("NSSTTA", "nonspecific st t abnormality")
DECODE_DICT["Dx_164867002"] = ("OldMI", "old myocardial infarction")
DECODE_DICT["Dx_10370003"] = ("PR", "pacing rhythm")
DECODE_DICT["Dx_251182009"] = ("VPVC", "paired ventricular premature complexes")
DECODE_DICT["Dx_282825002"] = ("PAF", "paroxysmal atrial fibrillation")
DECODE_DICT["Dx_67198005"] = ("PSVT", "paroxysmal supraventricular tachycardia")
DECODE_DICT["Dx_425856008"] = ("PVT", "paroxysmal ventricular tachycardia")
DECODE_DICT["Dx_284470004"] = ("PAC", "premature atrial contraction")
DECODE_DICT["Dx_427172004"] = ("PVC", "premature ventricular contractions")
DECODE_DICT["Dx_17338001"] = ("VPB", "ventricular premature beats")
DECODE_DICT["Dx_164947007"] = ("LPR", "prolonged pr interval")
DECODE_DICT["Dx_111975006"] = ("LQT", "prolonged qt interval")
DECODE_DICT["Dx_164917005"] = ("QAb", "qwave abnormal")
DECODE_DICT["Dx_164921003"] = ("RAb", "r wave abnormal")
DECODE_DICT["Dx_314208002"] = ("RAF", "rapid atrial fibrillation")
DECODE_DICT["Dx_253339007"] = ("RAAb", "right atrial abnormality")
DECODE_DICT["Dx_446358003"] = ("RAH", "right atrial hypertrophy")
DECODE_DICT["Dx_47665007"] = ("RAD", "right axis deviation")
DECODE_DICT["Dx_59118001"] = ("RBBB", "right bundle branch block")
DECODE_DICT["Dx_89792004"] = ("RVH", "right ventricular hypertrophy")
DECODE_DICT["Dx_55930002"] = ("STC", "s t changes")
DECODE_DICT["Dx_49578007"] = ("SPRI", "shortened pr interval")
DECODE_DICT["Dx_65778007"] = ("SAB", "sinoatrial block")
DECODE_DICT["Dx_427393009"] = ("SA", "sinus arrhythmia")
DECODE_DICT["Dx_426177001"] = ("SB", "sinus bradycardia")
DECODE_DICT["Dx_60423000"] = ("SND", "sinus node dysfunction")
DECODE_DICT["Dx_426783006"] = ("NSR", "sinus rhythm")
DECODE_DICT["Dx_427084000"] = ("STach", "sinus tachycardia")
DECODE_DICT["Dx_429622005"] = ("STD", "st depression")
DECODE_DICT["Dx_164931005"] = ("STE", "st elevation")
DECODE_DICT["Dx_164930006"] = ("STIAb", "st interval abnormal")
DECODE_DICT["Dx_251168009"] = ("SVB", "supraventricular bigeminy")
DECODE_DICT["Dx_63593006"] = ("SVPB", "supraventricular premature beats")
DECODE_DICT["Dx_426761007"] = ("SVT", "supraventricular tachycardia")
DECODE_DICT["Dx_251139008"] = ("ALR", "suspect arm ecg leads reversed")
DECODE_DICT["Dx_164934002"] = ("TAb", "t wave abnormal")
DECODE_DICT["Dx_59931005"] = ("TInv", "t wave inversion")
DECODE_DICT["Dx_266257000"] = ("TIA", "transient ischemic attack")
DECODE_DICT["Dx_164937009"] = ("UAb", "u wave abnormal")
DECODE_DICT["Dx_11157007"] = ("VBig", "ventricular bigeminy")
DECODE_DICT["Dx_164884008"] = ("VEB", "ventricular ectopics")
DECODE_DICT["Dx_75532003"] = ("VEsB", "ventricular escape beat")
DECODE_DICT["Dx_81898007"] = ("VEsR", "ventricular escape rhythm")
DECODE_DICT["Dx_164896001"] = ("VF", "ventricular fibrillation")
DECODE_DICT["Dx_111288001"] = ("VFL", "ventricular flutter")
DECODE_DICT["Dx_266249003"] = ("VH", "ventricular hypertrophy")
DECODE_DICT["Dx_251266004"] = ("VPP", "ventricular pacing pattern")
DECODE_DICT["Dx_195060002"] = ("VPEx", "ventricular pre excitation")
DECODE_DICT["Dx_164895002"] = ("VTach", "ventricular tachycardia")
DECODE_DICT["Dx_251180001"] = ("VTrig", "ventricular trigeminy")
DECODE_DICT["Dx_195101003"] = ("WAP", "wandering atrial pacemaker")
DECODE_DICT["Dx_74390002"] = ("WPW", "wolff parkinson white pattern")


def _load_df(data_path):
    ecgs = glob(f'{data_path}/*/*.hea')
    df = pd.DataFrame(ecgs)
    df.columns = ['ecg_file', ]
    df['label'] = df['ecg_file'].apply(lambda x: load_dxs(x)) 
    df['patient_id'] = range(df.shape[0])
    df = df[['ecg_file', 'patient_id', 'label']]
    return df


def _load_ecg(ecg_file):
    file = os.path.splitext(ecg_file)[0]
    record = wfdb.io.rdrecord(file)
    ecg = record.p_signal.T.astype('float32')
    leads = tuple(record.sig_name)
    sr = record.fs
    ecg[np.isnan(ecg)] = 0.0
    return ecg, leads, sr


def load_ann(file):
    ann = dict()
    with open(file, 'rt') as f:
        lines = f.readlines()

    num_leads = int(lines.pop(0).split()[1])

    leads = list()
    for i in range(num_leads):
        leads.append(lines.pop(0).split()[-1])

    ann['num_leads'] = num_leads
    
    for index, lead in enumerate(leads):
        ann[f'lead_{lead}'] = index
#     ann['leads'] = np.array(leads)

    for line in lines:
        if line.startswith('# Age'):
            ann['age'] = float(line.split(' ')[-1].strip())
        if line.startswith('# Sex'):
            ann['sex'] = line.split(' ')[-1].strip()
        if line.startswith('# Dx'):
            diagnoses = line.split()[-1].strip()
            if diagnoses != '':
                for diagnosis in diagnoses.split(','):
                    if diagnosis != '':
                        ann[f'Dx_{diagnosis.strip()}'] = 1
    return ann


def load_dxs(file):
    ann = load_ann(file)
    code_list = [DECODE_DICT[key][1] for key, val in ann.items() if key in DECODE_DICT]
    if len(code_list) == 1 and code_list[0] == 'sinus rhythm':
        code_list = ['normal ecg', ]
    else:
        code_list = [code for code in code_list if code != 'sinus rhythm']
    return ', '.join(code_list)
