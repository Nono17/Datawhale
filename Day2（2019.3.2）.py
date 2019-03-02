#!/usr/bin/env python
# coding: utf-8

# # Chapter 3

# In[1]:


tup = 4, 5, 6


# In[2]:


tup


# In[3]:


nested_tup = (4, 5, 6), (7, 8)


# In[4]:


nested_tup


# In[5]:


tuple([4, 0, 2])


# In[6]:


tup = tuple('string')


# In[7]:


tup


# In[8]:


tup[0]


# In[9]:


tup = tuple(['foo', [1, 2], True])


# In[10]:


tup[2] = False


# In[12]:


tup[1].append(3)


# In[13]:


tup


# In[14]:


(4, None, 'foo') + (6, 0) + ('bar',)


# In[15]:


('foo', 'bar') * 4


# In[16]:


tup = (4, 5, 6)


# In[17]:


a, b, c = tup


# In[18]:


b


# In[19]:


tup = 4, 5, (6, 7)


# In[20]:


a, b, (c, d) = tup


# In[21]:


d


# In[22]:


a, b = 1, 2


# In[23]:


a


# In[24]:


b


# In[25]:


b, a = a, b


# In[26]:


a


# In[27]:


b


# In[28]:


seq = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]


# In[29]:


for a, b, c in seq:
    print('a = {0}, b = {1}, c = {2}'.format(a, b, c))


# In[30]:


values = 1, 2, 3, 4, 5


# In[31]:


a, b, *rest = values


# In[32]:


a, b


# In[33]:


rest


# In[34]:


a, b, *_ = values


# In[35]:


a = (1, 2, 2, 2, 3, 4, 2)


# In[36]:


a.count(2)


# In[37]:


a_list = [2, 3, 7, None]


# In[38]:


tup = ('foo', 'bar', 'baz')


# In[39]:


b_list = list(tup)


# In[40]:


b_list


# In[41]:


b_list[1] = 'peekaboo'


# In[42]:


b_list


# In[45]:


gen = range(10)


# In[46]:


gen


# In[47]:


list(gen)


# In[48]:


b_list.append('dwarf')


# In[49]:


b_list


# In[50]:


b_list.insert(1, 'red')


# In[51]:


b_list


# In[52]:


b_list.pop(2)


# In[53]:


b_list


# In[54]:


b_list.append('foo')


# In[55]:


b_list


# In[56]:


b_list.remove('foo')


# In[57]:


b_list


# In[58]:


'dwarf' in b_list


# In[59]:


'dwarf' not in b_list


# In[60]:


[4, None, 'foo'] + [7, 8, (2, 3)]


# In[61]:


x = [4, None, 'foo']


# In[62]:


x.extend([7, 8, (2, 3)])


# In[63]:


x


# In[64]:


everything = []


# In[65]:


a = [7, 2, 5, 1, 3]


# In[66]:


a.sort()


# In[67]:


a


# In[68]:


b = ['saw', 'small', 'He', 'foxes', 'six']


# In[69]:


b.sort(key=len)


# In[70]:


b


# In[71]:


import bisect


# In[72]:


c = [1, 2, 2, 2, 3, 4, 7]


# In[73]:


bisect.bisect(c, 2)


# In[74]:


bisect.bisect(c, 5)


# In[75]:


bisect.insort(c, 6)


# In[76]:


c


# In[77]:


seq = [7, 2, 3, 7, 5, 6, 0, 1]


# In[78]:


seq[1:5]


# In[79]:


seq[3:4] = [6, 3]


# In[80]:


seq


# In[81]:


seq[:5]


# In[82]:


seq[3:]


# In[83]:


seq[-4:]


# In[84]:


seq[-6:-2]


# In[85]:


seq[::2]


# In[86]:


seq[::-1]


# In[87]:


some_list = ['foo', 'bar', 'baz']


# In[88]:


mapping = {}


# In[89]:


for i, v in enumerate(some_list):
    mapping[v] = i


# In[90]:


mapping


# In[91]:


sorted([7, 1, 2, 6, 0, 3, 2])


# In[92]:


sorted('horse race')


# In[93]:


seq1 = ['foo', 'bar', 'baz']


# In[94]:


seq2 = ['one', 'two', 'three']


# In[95]:


zipped = zip(seq1, seq2)


# In[96]:


list(zipped)


# In[97]:


seq3 = [False, True]


# In[98]:


list(zip(seq1, seq2, seq3))


# In[99]:


for i, (a, b) in enumerate(zip(seq1, seq2)):
    print('{0}: {1}, {2}'.format(i, a, b))


# In[100]:


pitchers = [('Nolan', 'Ryan'), ('Roger', 'Clemens'), ('Schilling', 'Curt')]


# In[101]:


first_names, last_names = zip(*pitchers)


# In[102]:


first_names


# In[103]:


last_names


# In[104]:


list(reversed(range(10)))


# In[105]:


empty_dict = {}


# In[106]:


d1 = { 'a': 'some value', 'b': [1, 2, 3, 4] }


# In[107]:


d1


# In[108]:


d1[7] = 'an integer'


# In[109]:


d1


# In[110]:


d1['b']


# In[111]:


'b' in d1


# In[112]:


d1[5] = 'some value'


# In[113]:


d1


# In[114]:


d1['dummy'] = 'another value'


# In[115]:


d1


# In[116]:


del d1[5]


# In[117]:


d1


# In[118]:


ret = d1.pop('dummy')


# In[119]:


ret


# In[120]:


d1


# In[121]:


list(d1.keys())


# In[122]:


list(d1.values())


# In[123]:


d1.update({ 'b': 'foo', 'c': 12 })


# In[124]:


d1


# In[125]:


mapping = dict(zip(range(5), reversed(range(5))))


# In[126]:


mapping


# In[127]:


words = ['apple', 'bat', 'bar', 'atom', 'book']


# In[128]:


by_letter = {}


# In[129]:


for word in words:
    letter = word[0]
    if letter not in by_letter:
        by_letter[letter] = [word]
    else:
        by_letter[letter].append(word)


# In[130]:


by_letter


# In[131]:


hash('string')


# In[132]:


hash((1, 2, (2, 3)))


# In[133]:


hash((1, 2, [2, 3]))


# In[134]:


d = {}


# In[135]:


d[tuple([1, 2, 3])] = 5


# In[136]:


set([2, 2, 2, 1, 3, 3])


# In[137]:


{2, 2, 2, 1, 3, 3}


# In[138]:


a = {1, 2, 3, 4, 5}


# In[139]:


b = {3, 4, 5, 6, 7, 8}


# In[140]:


a.union(b)


# In[141]:


a | b


# In[142]:


a.intersection(b)


# In[143]:


a & b


# In[144]:


c = a.copy()


# In[145]:


c |= b


# In[146]:


c


# In[147]:


d = a.copy()


# In[148]:


d &= b


# In[149]:


d


# In[150]:


my_data = [1, 2, 3, 4]


# In[151]:


my_set = {tuple(my_data)}


# In[152]:


my_set


# In[153]:


a_set = { 1, 2, 3, 4, 5 }


# In[154]:


{ 1, 2, 3 }.issubset(a_set)


# In[155]:


a_set.issubset({ 1, 2, 3 })


# In[156]:


{ 1, 2, 3 } == { 3, 2, 1 }


# In[157]:


strings = ['a', 'as', 'bat', 'car', 'dove', 'python']


# In[158]:


[x.upper() for x in strings if len(x) > 2]


# In[159]:


unique_lengths = { len(x) for x in strings }


# In[160]:


unique_lengths


# In[161]:


set(map(len, strings))


# In[162]:


loc_mapping = { val: index for index, val in enumerate(strings) }


# In[163]:


loc_mapping


# In[164]:


all_data = [['John', 'Emily', 'Michael', 'Mary', 'Steven'], ['Maria', 'Juan', 'Javier', 'Natalia', 'Pilar']]


# In[165]:


result = [ name for names in all_data for name in names if name.count('e') >= 2 ]


# In[166]:


result


# In[167]:


some_tuples = [ (1, 2, 3), (4, 5, 6), (7, 8, 9) ]


# In[168]:


flattened = [ x for tup in some_tuples for x in tup ]


# In[169]:


flattened


# In[170]:


[ [ x for x in tup ] for tup in some_tuples ]


# In[171]:


a = None


# In[172]:


def bind_a_variable():
    global a
    a = []
bind_a_variable()


# In[173]:


print(a)


# In[174]:


states = [ '    Alabama  ', 'Georgial', 'Georgia', 'georgia', 'Fl0rIda', 'south  carolina##', 'West virginia?' ]


# In[175]:


import re


# In[176]:


def clean_strings(strings):
    result = []
    for value in strings:
        value = value.strip()
        value = re.sub('[!#?]', '', value)
        value = value.title()
        result.append(value)
    return result


# In[177]:


clean_strings(states)


# In[178]:


def remove_punctuation(value):
    return re.sub('[!#?]', '', value)


# In[179]:


clean_ops = [str.strip, remove_punctuation, str.title]


# In[180]:


def clean_strings(strings):
    result = []
    for value in strings:
        for function in clean_ops:
            value = function(value)
        result.append(value)
    return result


# In[181]:


clean_strings([ '    Alabama  ', 'Georgial', 'Georgia', 'georgia', 'Fl0rIda', 'south  carolina##', 'West virginia?' ])


# In[182]:


for x in map(remove_punctuation, states):
    print(x)


# In[183]:


strings = ['foo', 'card', 'bar', 'aaaa', 'abab']


# In[184]:


strings.sort(key=lambda x: len(set(list(x))))


# In[185]:


strings


# In[186]:


some_dict = { 'a': 1, 'b': 2, 'c': 3 }


# In[187]:


for key in some_dict:
    print(key)


# In[188]:


dict_iterator = iter(some_dict)


# In[189]:


dict_iterator


# In[190]:


list(dict_iterator)


# In[191]:


def squares(n=10):
    print('Generating squares from i to {0}'.format(n**2))
    for i in range(1, n + 1):
        yield i ** 2


# In[192]:


gen = squares()


# In[193]:


gen


# In[194]:


for x in gen:
    print(x, end=" ")


# In[195]:


gen = (x ** 2 for x in range(10))


# In[196]:


gen


# In[197]:


sum(x ** 2 for x in range(5))


# In[198]:


dict((i, i ** 2) for i in range(5))


# In[199]:


import itertools


# In[200]:


first_letter = lambda x: x[0]


# In[201]:


names = ['Alan', 'Adam', 'Wes', 'Will', 'Albert', 'Steven']


# In[202]:


for letter, names in itertools.groupby(names, first_letter):
    print(letter, list(names))


# In[204]:


float('1.2345')


# In[205]:


float('something')


# In[206]:


def attempt_float(x):
    try:
        return float(x)
    except:
        return x


# In[207]:


attempt_float('1.2345')


# In[208]:


attempt_float('something')


# In[209]:


float(( 1, 2 ))


# In[210]:


def attempt_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return x


# In[211]:


attempt_float(( 1, 2 ))


# # Chapter 4

# In[214]:


import numpy as np


# In[215]:


my_arr = np.arange(1000000)


# In[216]:


my_list = list(range(1000000))


# In[217]:


get_ipython().run_line_magic('time', 'for _ in range(10): my_arr2 = my_arr * 2')


# In[218]:


get_ipython().run_line_magic('time', 'for _ in range(10): my_list2 = [ x * 2 for x in my_list ]')


# In[219]:


data = np.random.randn(2, 3)


# In[220]:


data


# In[221]:


data * 10


# In[222]:


data + data


# In[223]:


data.shape


# In[224]:


data.dtype


# In[225]:


data1 = [6, 7.5, 8, 0, 1]


# In[226]:


arr1 = np.array(data1)


# In[227]:


arr1


# In[228]:


data2 = [ [ 1, 2, 3, 4 ], [ 5, 6, 7, 8 ] ]


# In[229]:


arr2 = np.array(data2)


# In[230]:


arr2


# In[231]:


arr2.ndim


# In[232]:


arr2.shape


# In[233]:


arr1.dtype


# In[234]:


arr2.dtype


# In[235]:


np.zeros(10)


# In[236]:


np.zeros(( 3, 6 ))


# In[237]:


np.empty(( 2, 3, 2 ))


# In[238]:


np.arange(15)


# In[239]:


arr1 = np.array([ 1, 2, 3 ], dtype=np.float64)


# In[240]:


arr2 = np.array([ 1, 2, 3 ], dtype=np.int32)


# In[241]:


arr1.dtype


# In[242]:


arr2.dtype


# In[243]:


arr = np.array([ 1, 2, 3, 4, 5 ])


# In[244]:


arr.dtype


# In[245]:


float_arr = arr.astype(np.float64)


# In[246]:


float_arr.dtype


# In[247]:


arr = np.array([ 3.7, -1.2, -2.6, .5, 12.9, 10.1 ])


# In[248]:


arr


# In[249]:


arr.astype(np.int32)


# In[250]:


numeric_strings = np.array([ '1.25', '-9.6', '42' ])


# In[251]:


numeric_strings.astype(float)


# In[252]:


int_array = np.arange(10)


# In[253]:


calibers = np.array([ .22, .270, .357, .380, .44, .50 ], dtype=np.float64)


# In[254]:


int_array.astype(calibers.dtype)


# In[255]:


empty_uint32 = np.empty(8, dtype='u4')


# In[256]:


empty_uint32


# In[257]:


arr = np.array([ [ 1., 2., 3. ], [ 4., 5., 6. ] ])


# In[258]:


arr


# In[259]:


arr * arr


# In[260]:


arr - arr


# In[261]:


1 / arr


# In[262]:


arr ** .5


# In[263]:


arr2 = np.array([ [ 0., 4., 1. ], [ 7., 2., 12. ] ])


# In[264]:


arr2


# In[265]:


arr2 > arr


# In[266]:


arr = np.arange(10)


# In[267]:


arr


# In[268]:


arr[5]


# In[269]:


arr[5:8]


# In[270]:


arr[5:8] = 12


# In[271]:


arr


# In[272]:


arr_slice = arr[5:8]


# In[273]:


arr_slice


# In[274]:


arr_slice[1] = 12345


# In[275]:


arr


# In[276]:


arr_slice[:] = 64


# In[277]:


arr


# In[278]:


arr2d = np.array([ [ 1., 2., 3. ], [ 4., 5., 6. ], [ 7., 8., 9. ] ])


# In[279]:


arr2d[2]


# In[280]:


arr2d[0][2]


# In[281]:


arr2d[0, 2]


# In[282]:


arr3d = np.array([ [ [ 1, 2, 3 ], [ 4, 5, 6 ] ], [ [ 7, 8, 9 ], [ 10, 11, 12] ] ])


# In[283]:


arr3d


# In[284]:


arr3d[0]


# In[285]:


old_values = arr3d[0].copy()


# In[286]:


arr3d[0] = 42


# In[287]:


arr3d


# In[288]:


arr3d[0] = old_values


# In[289]:


arr3d


# In[290]:


arr3d[1, 0]


# In[291]:


x = arr3d[1]


# In[292]:


x


# In[293]:


x[0]


# In[294]:


arr


# In[295]:


arr[1:6]


# In[296]:


arr2d


# In[297]:


arr2d[:2]


# In[298]:


arr2d[:2, 1:]


# In[299]:


arr2d[1, :2]


# In[300]:


arr2d[:2, 2]


# In[301]:


arr2d[:, :1]


# In[302]:


arr2d[:2, 1:] = 0


# In[303]:


arr2d


# In[304]:


names = np.array([ 'Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe' ])


# In[305]:


data = np.random.randn(7, 4)


# In[306]:


names


# In[307]:


data


# In[308]:


names == 'Bob'


# In[309]:


data[names == 'Bob']


# In[310]:


data[names == 'Bob', 2:]


# In[311]:


data[names == 'Bob', 3]


# In[312]:


names != 'Bob'


# In[313]:


data[~(names == 'Bob')]


# In[314]:


cond = names == 'Bob'


# In[315]:


data[~cond]


# In[316]:


mask = (names == 'Bob') | (names == 'Will')


# In[317]:


mask


# In[318]:


data[mask]


# In[319]:


data[data < 0] = 0


# In[320]:


data


# In[321]:


data[names != 'Joe'] = 7


# In[322]:


data


# In[323]:


arr = np.empty(( 8, 4 ))


# In[324]:


for i in range(8):
    arr[i] = i


# In[325]:


arr


# In[326]:


arr[[ 4, 3, 0, 6 ]]


# In[327]:


arr[[ -3, -5, -7 ]]


# In[328]:


arr = np.arange(32).reshape(( 8, 4 ))


# In[329]:


arr


# In[330]:


arr[[ 1, 5, 7, 2], [ 0, 3, 1, 2 ]]


# In[331]:


arr = np.arange(15).reshape(( 3, 5 ))


# In[332]:


arr


# In[333]:


arr.T


# In[334]:


arr = np.random.randn(6, 3)


# In[335]:


arr


# In[336]:


np.dot(arr.T, arr)


# In[337]:


arr = np.arange(16).reshape(( 2, 2, 4 ))


# In[338]:


arr


# In[339]:


arr.transpose(( 1, 0, 2 ))


# In[340]:


arr


# In[341]:


arr.swapaxes(1, 2)


# In[342]:


arr = np.arange(10)


# In[343]:


arr


# In[344]:


np.sqrt(arr)


# In[345]:


np.exp(arr)


# In[346]:


x = np.random.randn(8)


# In[347]:


y = np.random.randn(8)


# In[348]:


x


# In[349]:


y


# In[350]:


np.maximum(x, y)


# In[351]:


arr = np.random.randn(7) * 5


# In[352]:


arr


# In[353]:


remainder, whole_part = np.modf(arr)


# In[354]:


remainder


# In[355]:


whole_part


# In[356]:


arr


# In[357]:


np.sqrt(arr)


# In[358]:


np.sqrt(arr, arr)


# In[359]:


arr


# In[360]:


points = np.arange(-5, 5, .01)


# In[361]:


xs, ys = np.meshgrid(points, points)


# In[362]:


ys


# In[363]:


z = np.sqrt(xs ** 2 + ys ** 2)


# In[364]:


z


# In[365]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[369]:


plt.imshow(z, cmap=plt.cm.gray)
plt.title('Image plot of $\sqrt{x^2 + y^2}$ for a grid of values')
plt.colorbar()


# In[370]:


xarr = np.array([ 1.1, 1.2, 1.3, 1.4, 1.5 ])


# In[371]:


yarr = np.array([ 2.1, 2.2, 2.3, 2.4, 2.5 ])


# In[372]:


cond = np.array([ True, False, True, True, False ])


# In[373]:


result = [ (x if c else y) for x, y, c in zip(xarr, yarr, cond) ]


# In[374]:


result


# In[375]:


result = np.where(cond, xarr, yarr)


# In[376]:


result


# In[377]:


arr = np.random.randn(4, 4)


# In[378]:


arr > 0


# In[379]:


np.where(arr > 0, 2, -2)


# In[380]:


np.where(arr > 0, 2, arr)


# In[382]:


arr = np.random.randn(5, 4)


# In[383]:


arr


# In[384]:


arr.mean()


# In[385]:


np.mean(arr)


# In[386]:


arr.sum()


# In[387]:


arr.mean(axis=1)


# In[388]:


arr.sum(axis=0)


# In[389]:


arr = np.array([ 0, 1, 2, 3, 4, 5, 6, 7 ])


# In[390]:


arr.cumsum()


# In[391]:


arr = np.array([ [ 0, 1, 2 ], [ 3, 4, 5 ], [ 6, 7, 8 ] ])


# In[392]:


arr


# In[393]:


arr.cumsum(axis=0)


# In[394]:


arr = np.random.randn(100)


# In[395]:


(arr > 0).sum()


# In[396]:


bools = np.array([ False, False, True, False ])


# In[397]:


bools.any()


# In[398]:


bools.all()


# In[399]:


arr = np.random.randn(6)


# In[400]:


arr


# In[401]:


arr.sort()


# In[402]:


arr


# In[404]:


arr = np.random.randn(5, 3)


# In[405]:


arr


# In[406]:


arr.sort(1)


# In[407]:


arr


# In[408]:


large_arr = np.random.randn(1000)


# In[409]:


large_arr.sort()


# In[410]:


large_arr[int(.05 * len(large_arr))]


# In[411]:


names


# In[412]:


np.unique(names)


# In[413]:


ints = np.array([ 3, 3, 3, 2, 2, 1, 1, 4, 4 ])


# In[414]:


np.unique(ints)


# In[415]:


sorted(set(names))


# In[416]:


values = np.array([ 6, 0, 0, 3, 2, 5, 6 ])


# In[417]:


np.in1d(values, [ 2, 3, 6 ])


# In[418]:


arr = np.arange(10)


# In[419]:


np.save('some_array', arr)


# In[420]:


np.load('some_array.npy')


# In[421]:


np.savez('array_archive.npz', a=arr, b=arr)


# In[422]:


arch = np.load('array_archive.npz')


# In[423]:


arch['b']


# In[424]:


np.savez_compressed('array_archive.npz', a=arr, b=arr)


# In[428]:


x = np.array([ [ 1., 2., 3. ], [ 4., 5., 6. ] ])


# In[429]:


y = np.array([ [ 6., 23. ], [ -1, 7 ], [ 8, 9 ] ])


# In[430]:


x


# In[431]:


y


# In[432]:


x.dot(y)


# In[433]:


np.dot(x, y)


# In[434]:


np.dot(x, np.ones(3))


# In[435]:


x @ np.ones(3)


# In[436]:


from numpy.linalg import inv, qr


# In[437]:


X = np.random.randn(5, 5)


# In[438]:


mat = X.T.dot(X)


# In[439]:


inv(mat)


# In[440]:


mat.dot(inv(mat))


# In[441]:


q, r = qr(mat)


# In[442]:


q


# In[443]:


r


# In[444]:


samples = np.random.normal(size=(4, 4))


# In[445]:


samples


# In[446]:


from random import normalvariate


# In[447]:


N = 1000000


# In[448]:


get_ipython().run_line_magic('timeit', 'samples = [ normalvariate(0, 1) for _ in range(N) ]')


# In[449]:


get_ipython().run_line_magic('timeit', 'np.random.normal(size=N)')


# In[450]:


np.random.seed(1234)


# In[451]:


rng = np.random.RandomState(1234)


# In[453]:


rng.randn(10)


# In[454]:


import random
position = 0
walk = [ position ]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)


# In[455]:


plt.plot(walk[:100])


# In[456]:


nsteps = 1000


# In[457]:


draws = np.random.randint(0, 2, size=nsteps)


# In[458]:


steps = np.where(draws > 0, 1, -1)


# In[459]:


walk = steps.cumsum()


# In[460]:


walk.min()


# In[461]:


walk.max()


# In[462]:


(np.abs(walk) >= 10).argmax()


# In[463]:


nwalks = 5000


# In[464]:


nsteps = 1000


# In[467]:


draws = np.random.randint(0, 2, size=(nwalks, nsteps))


# In[468]:


steps = np.where(draws > 0, 1, -1)


# In[469]:


walks = steps.cumsum(1)


# In[470]:


walks


# In[471]:


walks.max()


# In[472]:


walks.min()


# In[475]:


hits30 = (np.abs(walks) >= 30).any(1)


# In[476]:


hits30


# In[477]:


hits30.sum()


# In[478]:


crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)


# In[479]:


crossing_times.mean()


# In[480]:


steps = np.random.normal(loc=0, scale=.25, size=(nwalks, nsteps))


# In[482]:


steps = np.where(steps >= 0, 1, -1)


# In[483]:


walks = steps.cumsum(1)


# In[484]:


walks


# In[485]:


walks.min()


# In[486]:


walks.max()


# In[495]:


hits60 = (np.abs(walks) >= 60).any(1)


# In[496]:


hits60.sum()


# In[497]:


(np.abs(walks[hits60]) >= 60.).argmax(1).mean()

