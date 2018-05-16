import os

mail='maildir/germany-c/_sent_mail/1022.'
file='from_chris.txt'
path_win='C:\Users\\thb0f6\Documents\Digitalization\DAND\\6.Machine Learning\ud120-projects\\text_learning'
path1 = os.path.join(path_win,file)
path2 = os.path.join(os.curdir,'text_learning',file)
path3 = os.path.join(os.pardir,'test.png')
path4=os.path.join(os.curdir,'text_learning',mail)
print mail[:-1]
print path4
print os.path.exists(path4)

file_to_read="C:\\Users\\thb0f6\\Documents\\Digitalization\\DAND\\6.Machine Learning\\maildir\\bailey-s\\sent_items\\1"
os.startfile(file_to_read)
