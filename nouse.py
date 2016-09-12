from synonymcount import synonymcount
syObj=synonymcount()
s="comprehend"
a=dict()
a.update({'hello':syObj.synValues(s)})
print a['hello']