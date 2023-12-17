import numpy as np
import matplotlib.pylab as plt
from  khmernltk import word_tokenize


unseenData = ["ប្រាកដណាស់! ភាពយន្តគឺជាទម្រង់កម្សាន្តដ៏ពេញនិយមមួយ ដែលបានទាក់ទាញទស្សនិកជនជុំវិញពិភពលោកអស់រយៈពេលជាងមួយសតវត្សមកហើយ។ ពួកវាមានច្រើនប្រភេទ រួមទាំងរឿងភាគ រឿងកំប្លែង សកម្មភាព ប្រឌិតបែបវិទ្យាសាស្ត្រ ភ័យរន្ធត់ និងច្រើនទៀត។ ភាពយន្តមានថាមពលក្នុងការដឹកជញ្ជូនអ្នកទស្សនាទៅកាន់ពិភពលោកផ្សេងៗគ្នា បង្កើតអារម្មណ៍ និងប្រាប់រឿងគួរឱ្យទាក់ទាញអារម្មណ៍។"]

sentences = ["កីឡាគឺជាទម្រង់នៃការហាត់ប្រាណចម្រុះ និងពេញនិយមដែលពាក់ព័ន្ធនឹងសកម្មភាពរៀបចំ និងប្រកួតប្រជែង។ វាអាចត្រូវបានលេងជាលក្ខណៈបុគ្គល ឬជាក្រុម ហើយជារឿយៗត្រូវបានកំណត់ដោយច្បាប់ ឬបទប្បញ្ញត្តិជាក់លាក់។ កីឡាអាចត្រូវបានចាត់ថ្នាក់យ៉ាងទូលំទូលាយទៅជាប្រភេទផ្សេងៗ រួមទាំងកីឡាជាក្រុម (ដូចជាបាល់ទាត់ បាល់បោះ និងបាល់ទះ) កីឡាបុគ្គល (ដូចជា វាយកូនបាល់ វាយកូនហ្គោល និងហែលទឹក) និងសកម្មភាពផ្សេងទៀតដែលប្រហែលជាមិនសមនឹងប្រភេទទាំងនេះ (ដូចជា កាយសម្ព័ន្ធ ជិះកង់ និងក្បាច់គុន)។", "ការអប់រំសំដៅលើដំណើរការនៃការទទួលបានចំណេះដឹង ជំនាញ គុណតម្លៃ និងអាកប្បកិរិយាប្រកបដោយរចនាសម្ព័ន្ធ និងជាប្រព័ន្ធ។ វាជាដំណើរការពេញមួយជីវិត និងបន្តដែលប្រព្រឹត្តទៅក្នុងការកំណត់ផ្លូវការ និងក្រៅផ្លូវការផ្សេងៗ។ ការអប់រំផ្លូវការជាធម្មតាកើតឡើងនៅក្នុងសាលារៀន មហាវិទ្យាល័យ និងសាកលវិទ្យាល័យ ដែលសិស្សធ្វើតាមកម្មវិធីសិក្សា ហើយត្រូវបានបង្រៀនដោយគ្រូ ឬសាស្រ្តាចារ្យ។ ម្យ៉ាងវិញទៀត ការអប់រំក្រៅផ្លូវការ កើតឡើងតាមរយៈបទពិសោធន៍ប្រចាំថ្ងៃ អន្តរកម្មជាមួយអ្នកដទៃ និងការរៀនសូត្រដោយខ្លួនឯង ។"]

all_words = []

for setence in sentences:
    for word in word_tokenize(setence):
        all_words.append(word)

set_all_words = list(set(all_words))

bag_word = []
unseen_bag_word = []

for sentence in sentences:
    bag_vector = [1 if word in word_tokenize(sentence) else 0 for word in set_all_words]
    bag_word.append(bag_vector)

unseen_bag_word = [1 if word in word_tokenize(unseenData[0]) else 0 for word in set_all_words]

bag_word = np.array(bag_word, dtype=np.int16)
unseen_bag_word = np.array(unseen_bag_word, dtype=np.int16)


xs = bag_word

ys = np.asarray([[0],
                 [1]
                 ])



ins = 110
outs = 1


def weight(input_x, output):
    wss = np.random.randn(input_x, output)
    return wss


ws = weight(ins, outs)



ers = []
for i in range(5000):
    yh = xs @ ws
    e = yh - ys
    e = np.sum(np.abs(e))
    if e < 0.05:
        print("found solution")
        break
    else:
        mutation = weight(ins, outs) * 0.09
        cw = ws + mutation
        yh = xs @ cw
        ce = yh - ys
        ce = np.sum(np.abs(ce))
        if ce < e:
            ws = cw
    ers.append(e)

print("Unseen DAta :", unseen_bag_word @ ws)
print("seen data :", xs @ ws)

# plt.figure(1)
# plt.plot(ers)
# plt.show()
