#!/usr/bin/env python
# coding: utf-8

# # The second Chapter

# In[7]:


import numpy as np


# In[11]:


data ={i : np.random.randn()for i in range (7)}


# In[12]:


data


# # Tab Completion

# In[13]:


an_apple = 27


# In[14]:


an_example = 42


# In[21]:


an_apple


# In[22]:


b=[1,2,3]


# In[23]:


b.append


# In[24]:


import datetime


# In[25]:


datetime.date


# # introspection

# In[26]:


b=[1,2,3]


# In[27]:


get_ipython().run_line_magic('pinfo', 'b')


# In[28]:


get_ipython().run_line_magic('pinfo', 'print')


# In[36]:


def add_numbers (a, b):
    """
    Add two numbers together
Returns
-------
the_sum :type of arguments

"""
return a + b


# In[37]:


get_ipython().run_line_magic('pinfo', 'add_numbers')


# In[40]:


get_ipython().run_line_magic('pinfo2', 'add_numbers')


# In[41]:


get_ipython().run_line_magic('psearch', 'np.*load*')


# # The%run Command

# In[42]:


def f(x,y,z):
    return(x+y)/z
a=5
b=6
c=7.5
result=f(a,b,c)


# In[43]:


get_ipython().run_line_magic('run', 'ipython_script_test.py')


# In[44]:


get_ipython().run_line_magic('load', 'ipython_script_test.py')


# In[46]:


get_ipython().run_line_magic('matplotlib', 'inline')


# import matplotlib.pyplot as plt
# plt.plot(np.random.randn(5).cumsum())

# In[47]:


result=[]
for line in file_handle:
    result.append(line.replace('foo','bar'))


# In[48]:


def append_element(some_list,element):
    some_list.append(element)


# In[49]:


data =[1,2,3]


# In[50]:


append_element(data,4)


# In[51]:


data


# In[52]:


a=5


# In[53]:


type(a)


# In[54]:


a='foo'


# In[55]:


type(a)


# In[56]:


a=4.5


# In[57]:


b=2


# In[59]:


print('a is {0},b is {1} '.format(type(a),type(b)))


# In[60]:


a/b


# In[61]:


a=5


# In[62]:


isinstance(a,int)


# In[63]:


a=5


# In[64]:


b=4.5


# In[66]:


isinstance(a,(int ,float))


# In[67]:


isinstance(b,(int,float))


# In[68]:


def isiterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


# In[69]:


isiterable('a string')


# In[70]:


isiterable([1,2,3])


# In[72]:


isiterable(5)


# In[73]:


a=[1,2,3]


# In[74]:


b=a


# In[75]:


a is b


# In[76]:


c=list(a)


# In[77]:


a is not c


# In[78]:


a_list=['foo',2,[4,5]]


# In[79]:


a_list[2]=(3,4)


# In[80]:


a_list


# In[81]:


a_tuple=(3,5,(4,5))


# In[82]:


a_tuple[1]='four'


# In[83]:


c='''
This is a longer string that
spans multiple lines
'''


# In[85]:


c.count('\n')


# In[86]:


s='python'


# In[87]:


list(s)


# In[88]:


s[:3]


# In[101]:


template=' {0:.2f} {1:s} are worth US${2:d}'


# In[102]:


template.format(4.5, 'Argentine Pesos',1)


# In[104]:


val="espanol"
val_utf8 =val.encode('utf-8')


# In[105]:


val_utf8


# In[107]:


type(val_utf8)


# In[113]:


def add_and_maybe_multiply(a,b,c=None):
    result =a+b
    if c is not None :
        result =result *c
    return result


# In[114]:


from datetime import datetime,date,time


# In[116]:


dt=datetime(2019,2,28,11,45)


# In[117]:


dt.day


# In[119]:


dt.minute


# In[122]:


dt.date()


# In[123]:


dt.time()


# In[125]:


dt.strftime('%m/%d/%Y/%H:%M')


# In[130]:


if x<0:
    print('It’s negative')
elif x==0:
    print('Equal to zero')
elif 0<x<5:
    print('Positive but smaller than 5')
else:
    print('Positive and larger than or equal to 5')


# In[131]:


sequence =[1,2,None,4,None,5]
total=0
for value in sequence:
    if value is None:
        continue
    total+=value


# In[134]:


sequence =[1,2,0,4,6,5,2,1]
total_until_5=0
for value in sequence:
    if value==5:
        break
    total_until_5 +=value


# In[135]:


for i in range(4):
    for j in range(4):
        if j>i:
            break
        print((i,j))


# In[136]:


range(10)


# In[137]:


list(range(10))


# In[138]:


list(range(0,20,2))


# In[139]:


list(range(5,0,-1))


# In[ ]:




