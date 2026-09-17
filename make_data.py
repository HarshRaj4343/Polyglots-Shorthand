"""Authored synthetic Hinglish benchmark. Split groups BEFORE augmentation."""


# Key idea: a seed and ALL its variants go to the same split, so the test set never contains a near-copy of a training sentence (no "leakage").

import json
import random
from pathlib import Path

SEEDS = {
'cancel_order': [
('u','bhai order cancel kar do please urgent meeting hai'),
('u','mujhe apni booking cancel karni hai'),
('u','kal wala order ab nahi chahiye cancel please'),
('n','itna late ho gaya cancel karo ab 😒'),
('u','galti se do order ho gaye ek cancel karna'),
('u','cancel request kaise bheju'),
('n','bekar service order cancel kar do 😡'),
('u','dispatch se pehle order rok do please'),
('u','aaj ka purchase cancel kar sakte ho kya'),
('n','bahut pareshan hu mera order cancel karo'),
('u','cancel option app mein kahan milega'),
('u','delivery mat bhejna order cancel karna hai'),
('u','plan change ho gaya cancel my order'),
('n','ab aur wait nahi karna cancel karo 😠'),
('u','is booking ko cancel karne mein help chahiye'),
('u','please order ki cancellation karwa do')],
'refund': [
('u','bhai refund kab tak aayega'),
('n','paise abhi tak wapas nahi aaye 😡'),
('u','refund ka status check kar do'),
('u','cancel ho gaya ab paise wapas chahiye'),
('n','refund ke liye itna wait kyun 😒'),
('u','mere bank mein refund kab credit hoga'),
('u','return ke baad amount wapas kaise milega'),
('n','refund reject kar diya bahut bekar service'),
('u','mujhe payment ka refund request karna hai'),
('u','paise wapas lene ka process batao'),
('n','do hafte ho gaye refund nahi mila'),
('u','wallet mein refund bhej sakte ho kya'),
('u','mera refund process hua ya pending hai'),
('n','amount wapas do yaar kitni baar bolu 😠'),
('u','refund ki expected date bata dena'),
('u','return pickup ho gaya payment wapas kab milega')],
 'track_order': [
('u','mera order kahan hai'),
('u','delivery kab tak hogi bhai'),
('u','parcel ka current location batao'),
('n','order late hai bahut pareshan hu 😡'),
('u','tracking link share kar do please'),
('u','aaj parcel aane wala hai kya'),
('u','shipment dispatch hua ya nahi'),
('n','delivery ke liye poora din wait kiya bekar'),
('u','order ka latest status kya hai'),
('u','delivery boy abhi kahan pahuncha'),
('u','expected delivery date bata sakte ho'),
('n','itni slow delivery kyun hai 😒'),
('u','package raste mein hai kya update dena'),
('u','mera courier kab pahunchega'),
('n','tracking update hi nahi hota ghatiya app'),
('u','please mere parcel ko track karke batao')],
 'not_received': [
('n','item delivered bol raha hai but mila hi nahi 😡'),
('u','app par delivered hai lekin parcel nahi mila'),
('u','delivery complete dikh rahi hai par receive nahi hua'),
('n','jhooth bol rahe delivered kuch mila nahi 😒'),
('u','courier ne delivered mark kiya ghar pe nahi aaya'),
('u','package kisi aur ko de diya mujhe nahi mila'),
('n','delivered message aaya par saman gayab hai bekar'),
('u','order receive nahi kiya lekin delivery done hai'),
('n','bina parcel diye delivered kar diya 😠'),
('u','gate pe check kiya parcel nahi hai app says delivered'),
('u','status delivered hai kya proof mil sakta hai mujhe nahi mila'),
('n','fake delivery update mera package mila hi nahi'),
('u','delivery confirmation aa gaya lekin saman nahi pahuncha'),
('u','mere paas parcel nahi hai tracking delivered bata raha'),
('n','delivered likh diya bina order diye bahut gussa aa raha'),
('u','neighbor ke paas bhi nahi hai par order delivered hai')],
 'damaged_item': [
('n','box khola toh item toota hua hai 😡'),
('u','parcel mein damaged product aaya hai'),
('n','screen cracked hai ghatiya packing'),
('u','bottle leak ho rahi hai replacement chahiye'),
('n','kapda phata hua bheja bahut bekar'),
('u','order receive hua lekin part broken hai'),
('n','dabba pura crush ho gaya andar saman toota 😒'),
('u','product par scratch hai exchange ho sakta hai'),
('u','charger damaged nikla replacement ka process batao'),
('n','quality kharab hai item pehle se tuta tha'),
('u','packet open tha aur item cracked hai'),
('n','broken piece bhej diya seriously 😠'),
('u','received product damaged hai photo kahan bheju'),
('u','headphone ka ek side toota hua mila'),
('n','itni bekar packaging sab kuch toot gaya'),
('u','delivered item mein crack hai replace karna hai')],
 'feedback': [
('p','bahut acchi service thank you'),
('p','mast experience tha yaar'),
('p','team ne kamaal ki help kari'),
('p','support ekdum zabardast hai'),
('p','badiya kaam kiya tumne'),
('p','service se bahut khush hu'),
('p','super helpful team dil jeet liya'),
('p','wah kya fantastic service hai'),
('n','bahut kharab experience tha'),
('n','support se bilkul khush nahi hu'),
('n','ye service acchi nahi hai'),
('n','help ke naam pe time waste kiya'),
('u','theek tha experience bas normal'),
('u','service average lagi mujhe'),
('u','na accha na bura normal experience'),
('u','bas feedback dena tha koi khas baat nahi')]
}

# Sentiment Categories: p = positive, n = negative, u = neutral.
# Intent Categories: cancel_order, refund, track_order, not_received, damaged_item, feedback.

# create realistic spelling variants of a message.

SHORT = {'kar':'kr','karo':'kro','karna':'krna','raha':'rha','rahe':'rhe',
         'nahi':'nhi','hai':'h','please':'plz','bahut':'bht','kahan':'kaha'}
# Replace every word that has a shorthand form, e.g. "nahi hai" -> "nhi h".
def shorthand(text):
    return ' '.join(SHORT.get(t,t) for t in text.split())

def variants(text):
    # Do not add/delete negation, emojis or sentiment-bearing punctuation.
    # Returns up to 3 unique versions: original, shorthand, and a misspelled
    # version ("ordr", "delivry", "servis"). dict.fromkeys removes duplicates
    # while keeping order.
    return list(dict.fromkeys([text, shorthand(text),
                              text.replace('order','ordr').replace('delivery','delivry').replace('service','servis')]))


def build(root):
    rng=random.Random(42)
    rows=[]
    for intent,seeds in SEEDS.items():
        assignments={}
        for sentiment in ('p','n','u'):
            ids=[i for i,(s,_) in enumerate(seeds) if s==sentiment]
            rng.shuffle(ids)
            nv=max(1,round(len(ids)*.18)) if len(ids)>=3 else 0
            nt=max(1,round(len(ids)*.22)) if len(ids)>=3 else 0
            for j,i in enumerate(ids):
                assignments[i]='validation' if j<nv else 'test' if j<nv+nt else 'train'
        for i,(s,text) in enumerate(seeds):
            group=f'{intent}-{i:02d}'
            forms=[(text,{'p':'positive','n':'negative','u':'neutral'}[s])]
            # Contrast examples are annotation assumptions, not universal emoji rules.
            # Positive feedback also gets a "😊" copy (positive) and a sarcastic
            # "😒" copy (negative) so the model learns emojis can flip tone.
            if intent=='feedback' and s=='p':
                forms += [(text+' 😊','positive'),(text+' 😒','negative')]
            # Add every spelling variant as its own row, all in the seed's split.
            for base,sentiment in forms:
                for text_variant in variants(base):
                    rows.append({'text':text_variant,'intent':intent,'sentiment':sentiment,
                                 'group':group,'split':assignments[i], 'language':'hi-en',
                                 'source':'authored_synthetic'})
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    for split in ('train','validation','test'):
        selected=[r for r in rows if r['split']==split]
        (root/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
    return rows

if __name__=='__main__':
    rows=build(Path(__file__).parent/'data')
    print(json.dumps({s:sum(r['split']==s for r in rows) for s in ('train','validation','test')},indent=2))
