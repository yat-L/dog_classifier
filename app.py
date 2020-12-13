#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastbook import *
from fastai.vision.widgets import *


# In[2]:


#!pip install voila


# In[10]:


#!jupyter serverextension enable voila --sys-prefix


# In[4]:


learn_inf = load_learner('export.pkl')


# In[5]:


show_upload = widgets.Output()
prediction_result = widgets.Label()


# In[6]:


upload_button = widgets.FileUpload()


# In[7]:


classify_button = widgets.Button(description='Classify')


# In[8]:


def on_click_classify(change):
    image = PILImage.create(upload_button.data[-1])
    show_upload.clear_output()
    with show_upload: display(image.to_thumb(128,128))
    prediction, index, proba = learn_inf.predict(image)
    prediction_result.value = f'Prediction: {prediction}; Probability: {proba[index]:.04f}'

classify_button.on_click(on_click_classify)


# In[9]:


VBox([widgets.Label('Select a picture of a golden retriever / shiba / bulldog / german shepherd / beagles .'),
      upload_button, classify_button, show_upload, prediction_result])


# In[ ]:





# In[ ]:




