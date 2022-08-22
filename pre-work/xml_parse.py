import xml.dom.minidom as xmldom
import os.path


def voc_xml_parse(xml_path):
    object_list = []
    domobj = xmldom.parse(xml_path)
    elementobj = domobj.documentElement
    folderobj = elementobj.getElementsByTagName("folder")[0]
    filenameobj = elementobj.getElementsByTagName("filename")[0]
    sourceobj = elementobj.getElementsByTagName("source")[0]
    #ownerobj = elementobj.getElementsByTagName("owner")[0]
    sizeobj = elementobj.getElementsByTagName("size")[0]
    # segmentedobj = elementobj.getElementsByTagName("segmented")[0]
    head = { 'folder':folderobj, 'filename':filenameobj, 'source':sourceobj,  'size':sizeobj}
    object_list = elementobj.getElementsByTagName("object")
    return head, object_list


def voc_xml_modify(modify_xml_path, head, object_list):
    dom = xmldom.Document()
    root = dom.createElement('annotation')
    dom.appendChild(root)
    for obj in head.values():
        root.appendChild(obj)
    for obj in object_list:
        root.appendChild((obj))
    with open(modify_xml_path, 'w', encoding='utf-8') as f:
        dom.writexml(f, addindent='\t',  newl='\n', encoding='utf-8')
    return
