import pandas as pd

MAIN_DATABSE = 'ptb_xl'
NORMAL_CLASS = 'normal ecg'

def remove_classes(classes_to_remove, captions):
    new = list()
    for caption in captions:
        classes = [class_.strip().lower() for class_ in caption.strip().split(',')]
        classes = [class_ for class_ in classes if class_ not in classes_to_remove]
        classes = ', '.join(classes)
        new.append(classes)
    return new
    

def fix_names(class_df, datasets, threshold=150):
    for index, rename_to in class_df['rename_to'].items():
        if not pd.isna(rename_to):
            rename_to_index = class_df[class_df['class_name'] == rename_to]
            assert rename_to_index.shape[0] == 1
            rename_to_index = rename_to_index.index.values[0]
            class_df.loc[rename_to_index, datasets] += class_df.loc[index, datasets]

    class_fixing_dict = class_df[~class_df['rename_to'].isna()].set_index('class_name')['rename_to'].to_dict()

    class_df = class_df[class_df['rename_to'].isna()]
    zeroshot_classes = class_df[class_df['status'] == 'zero-shot']
    class_df = class_df[class_df['status'] != 'zero-shot']

    train_classes_mask = (class_df[MAIN_DATABSE] >= threshold) & (class_df['class_name'] != NORMAL_CLASS)
    train_classes = class_df[train_classes_mask]
    noteval_classes = class_df[~train_classes_mask]
    return train_classes, noteval_classes, zeroshot_classes, class_fixing_dict


def prepare_df(df, train_classes, zeroshot_classes, class_fixing_dict):
    def fix_caption(caption):
        if caption == '':
            return NORMAL_CLASS
        split = caption.lower().strip().split(', ')
        fixed_caption = list()
        for class_ in split:
            class_ = class_.strip()
            if class_ in class_fixing_dict:
                class_ = class_fixing_dict[class_]
            fixed_caption.append(class_)

        if len(fixed_caption) > 1 and NORMAL_CLASS in fixed_caption:
            fixed_caption.remove(NORMAL_CLASS)
        return ', '.join(fixed_caption)

    df['fixed_label'] = df['label'].apply(fix_caption)
    for class_ in train_classes['class_name'].values:
        df[class_] = df['fixed_label'].apply(lambda x: class_ in x)

    for class_ in zeroshot_classes['class_name'].values:
        df[class_] = df['fixed_label'].apply(lambda x: class_ in x)   

    df['train_label'] = remove_classes(zeroshot_classes['class_name'].to_list(), df['fixed_label'].to_list())
    df['train_label'] = df['train_label'].apply(lambda x: x if x != '' else NORMAL_CLASS)
    return df
