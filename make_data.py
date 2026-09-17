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

# Enrichment (v2). The original seeds had positive sentiment ONLY inside
# `feedback` (8 seeds), so the model learned "positive == feedback" and test
# recall for positive was 0. These extra seeds add positive tone to EVERY
# intent, harder negatives/neutrals, negation ("accha nahi laga"), English-heavy
# code-mixing and longer multi-clause messages where context decides the label.
MORE_SEEDS = {
'cancel_order': [
('p','thank you team bas mera order cancel kar do please 😊'),
('p','aap log bahut helpful ho ek order cancel karna tha'),
('p','shukriya jaldi reply ke liye booking cancel kar dijiye'),
('p','great service hamesha is baar order cancel karna padega'),
('p','love this app 😍 bas galat size order ho gaya cancel kar do'),
('p','bahut accha support mila cancellation ho gaya thanks'),
('p','itni fast cancellation ke liye dil se dhanyavaad 🙏'),
('p','awesome team order cancel karne mein help karo please'),
('p','cancel ho gaya bina kisi hassle ke superb experience'),
('p','aapka app mast hai bas ye wala order cancel kar do'),
('p','khushi hui itni aasaan cancellation dekh ke thank you'),
('n','teen baar cancel dabaya kuch nahi hua ghatiya app 😡'),
('n','cancel karne ke paise kaat liye ye toh loot hai'),
('n','order cancel kyun nahi ho raha bakwaas system'),
('n','customer care phone nahi uthata cancel kaise karu 😤'),
('n','bina puche order confirm kar diya turant cancel karo bekar'),
('n','sabse worst app cancel button hi kaam nahi karta'),
('n','ye order chahiye hi nahi tha cancel karo bahut gussa aa raha hai'),
('n','cancellation reject kyun kiya faltu policy hai'),
('u','order number 4521 cancel karna hai'),
('u','kya shipped order bhi cancel ho sakta hai'),
('u','cash on delivery wala order cancel karna hai'),
('u','cancel karne par koi charge lagega kya'),
('u','subscription cancel karni hai next month se')],
'refund': [
('p','refund aa gaya account mein thank you so much 😊'),
('p','itni jaldi refund mil gaya bahut badiya'),
('p','paise wapas aa gaye great support team'),
('p','refund process ekdum smooth tha khush hu'),
('p','shukriya refund do din mein credit ho gaya'),
('p','aap logon ne refund jaldi kar diya dil se thanks 🙏'),
('p','wallet mein refund turant mila awesome service'),
('p','refund ke liye itni acchi help mili superb'),
('p','agent bahut helpful tha refund status clear kar diya'),
('p','love the quick refund ekdum mast'),
('n','refund ke naam pe sirf jhooth bolte ho 😡'),
('n','ek mahina ho gaya paise nahi aaye bakwaas'),
('n','refund amount kam kyun bheja ye cheating hai'),
('n','har baar bolte ho teen din mein aayega faltu log'),
('n','mera paisa kha gaye kya refund do abhi 😤'),
('n','refund pending pending pending bahut ghatiya service'),
('u','refund upi mein aayega ya card mein'),
('u','partial refund ka calculation samjha do'),
('u','refund ka reference number chahiye'),
('u','cod order ka refund kaise milega')],
'track_order': [
('p','order time pe aa raha hai thanks for the update 😊'),
('p','tracking bahut accurate hai great app'),
('p','delivery boy bahut polite tha abhi location share kar do'),
('p','itni fast shipping wah kab tak pahunchega'),
('p','thank you tracking link mil gaya super helpful'),
('p','aapka tracking feature mast hai parcel kal aayega na'),
('p','shukriya update ke liye order ka status dekh liya'),
('p','love it order ek din pehle hi dispatch ho gaya 😍'),
('p','bahut accha laga live tracking dekh ke'),
('p','excellent service bas delivery time confirm kar do'),
('n','paanch din se out for delivery dikha raha hai bakwaas 😡'),
('n','tracking page hamesha error deta hai ghatiya'),
('n','kal aana tha abhi tak nahi aaya koi jawab nahi deta'),
('n','courier wale ka number band hai bahut frustrating 😤'),
('n','har din nayi delivery date jhoothe log'),
('n','order kahan gaya kisi ko pata hi nahi worst'),
('n','bahut der ho gayi status batao warna complaint karunga'),
('u','awb number se track kaise karu'),
('u','order shipped hua hai kya'),
('u','kaunsa courier partner hai mere order ka')],
'not_received': [
('p','aap log hamesha reliable ho is baar parcel delivered dikha ke nahi mila check kar do 🙏'),
('p','thank you quick reply ke liye order delivered hai lekin mujhe mila nahi'),
('p','support team bahut acchi hai ek parcel receive nahi hua usme help karo'),
('p','shukriya aapne turant complaint le li parcel nahi mila tha'),
('p','missing parcel neighbour ke paas mil gaya thank you 😊'),
('p','delivered dikha raha tha par mila nahi aapne replacement bhej diya thanks a lot'),
('n','delivered likha hai par ghar pe kuch nahi aaya fraud hai kya 😡'),
('n','teesri baar parcel gayab ghatiya delivery'),
('n','guard bhi bol raha koi parcel nahi aaya bakwaas service'),
('n','order mila nahi aur complaint bhi close kar di 😤'),
('n','delivery boy ne fake otp daal diya saman mila hi nahi'),
('u','delivered status aaya but parcel receive nahi hua'),
('u','parcel receive nahi hua delivery otp bhi nahi aaya'),
('u','kisi ne mera parcel receive kiya kya mere paas nahi aaya'),
('u','delivered bola hai lekin mere address pe nahi pahuncha')],
'damaged_item': [
('p','item thoda damage tha par replacement fast mila thank you 😊'),
('p','packaging usually acchi hoti hai is baar product toota aaya replace kar do please'),
('p','aapki team ne damaged phone ka turant replacement diya great'),
('p','shukriya broken glass ki photo bhej di aap log bahut helpful ho'),
('p','cracked screen tha lekin support superb tha exchange ho gaya'),
('p','love your service bas mug toota hua mila replace kar do 🙏'),
('n','naya phone aur screen pe crack ye kaisi quality hai 😡'),
('n','dusri baar bhi toota hua item bheja bakwaas'),
('n','kharab product bheja aur return bhi nahi le rahe bahut gussa'),
('n','shoes phate hue the worst packaging ever 😤'),
('n','expiry wala saman bheja sab kharab nikla'),
('u','mixer grinder ka lid damaged aaya hai'),
('u','product dented hai return request dalni hai'),
('u','bottle ka cap tuta hua hai'),
('u','laptop pe dent hai replacement policy kya hai')],
'feedback': [
('p','delivery boy ka behaviour bahut accha tha'),
('p','app use karna bahut easy hai love it ❤️'),
('p','best shopping experience ever thank you'),
('p','customer care ne dil khush kar diya'),
('p','packaging bahut sundar thi great job'),
('p','aapki service se hamesha satisfied rehta hu'),
('p','five star service keep it up 👍'),
('p','bahut shukriya itne acche support ke liye 🙏'),
('p','quality ekdum top class thi maza aa gaya'),
('p','fast delivery aur acche daam superb'),
('n','app bahut slow hai bilkul pasand nahi aaya'),
('n','customer care ka behaviour rude tha 😡'),
('n','worst experience dobara order nahi karunga'),
('n','support wale sirf script padhte hain useless'),
('n','itna bura experience kabhi nahi hua'),
('n','service accha nahi laga kaafi disappointed hu'),
('n','quality ghatiya thi paisa barbaad 😤'),
('u','app ka naya update theek hai'),
('u','delivery time normal tha'),
('u','app ke liye kuch suggestions dene the'),
('u','packaging theek thi kuch khaas nahi'),
('u','bas rating dena tha'),
('u','experience okay tha')],
}
for _intent, _extra in MORE_SEEDS.items():
    SEEDS[_intent] = SEEDS[_intent] + _extra

# Sentiment Categories: p = positive, n = negative, u = neutral.
# Intent Categories: cancel_order, refund, track_order, not_received, damaged_item, feedback.

# create realistic spelling variants of a message.

SHORT = {'kar':'kr','karo':'kro','karna':'krna','raha':'rha','rahe':'rhe',
         'nahi':'nhi','hai':'h','please':'plz','bahut':'bht','kahan':'kaha'}
# Replace every word that has a shorthand form, e.g. "nahi hai" -> "nhi h".
def shorthand(text):
    return ' '.join(SHORT.get(t,t) for t in text.split())

# Alternative spellings / code-switches. Deliberately DIFFERENT from the stress
# test's misspellings in solution.py, so that stress test stays unseen.
SPELLING = {'order':'ordr','delivery':'delivry','service':'servis','parcel':'parcl',
            'bahut':'bohot','accha':'acha','thank':'thnk','paise':'money',
            'jaldi':'fast','mujhe':'mujhko','chahiye':'chaiye','kab':'when'}
# Tone-free openers people type before the real message.
OPENERS = ['bhai','hello team','hi','sir','dekhiye','ji']

def respell(text):
    return ' '.join(SPELLING.get(t,t) for t in text.split())

def variants(text, rng=None):
    # Do not add/delete negation, emojis or sentiment-bearing punctuation.
    # Returns up to 5 unique versions: original, shorthand, respelled,
    # shorthand+respelled and (when an rng is given) a tone-free opener.
    # dict.fromkeys removes duplicates while keeping order.
    forms = [text, shorthand(text), respell(text), shorthand(respell(text))]
    if rng is not None:
        forms.append(rng.choice(OPENERS) + ' ' + text)
    return list(dict.fromkeys(forms))


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
                for text_variant in variants(base, rng):
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
