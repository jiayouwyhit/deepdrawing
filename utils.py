
from xml.dom.minidom import Document
import cairosvg
import os
import numpy as np

def printtensor(X):
    XL = X.tolist()
    for i in range(len(XL)):
        print("line : %d "%(i))
        print(XL[i])
def EuclideanDistances(A, B):
    BT = B.T
    vecProd = np.dot(A,BT)
    SqA =  A**2
    sumSqA = np.sum(SqA,1)
    sumSqAExt = np.expand_dims(sumSqA,1)
    sumSqAEx = np.repeat(sumSqAExt,vecProd.shape[0],axis=1)
    sumSqBEx = sumSqAEx.T
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    SqED[SqED<1e-16]=1e-16 
    ED = np.sqrt(SqED)
    return ED

def node_occlusion_calculate(pos,threshold):
    ### Calculate how many edges length < threshold 
    dist = EuclideanDistances(pos,pos)
    value = np.sum(np.int64(dist<threshold)) - pos.shape[0]
    return value/2
## Visualize Data
def visualize(y,ori,name,test_folder,mode=1,text=None):
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)
    # create svg document 
    #mode = 1
    if y is None:
        mode = 0
    width = float(ori["width"])
    height = float(ori["height"])
    doc = Document()  
    svg = doc.createElement('svg') 
    svg.setAttribute('xmlns',"http://www.w3.org/2000/svg")
    svg.setAttribute('width',str(width))
    svg.setAttribute('height',str(height))
    svg.setAttribute('xmlns:xlink',"http://www.w3.org/1999/xlink")
    
    doc.appendChild(svg)
    style = doc.createElement('style')
    svg.appendChild(style)
    style.setAttribute('xmlns',"http://www.w3.org/1999/xhtml")
    style.setAttribute('type',"text/css")
    

    style_text = doc.createTextNode(
    '''
    .links line {
      stroke: #999;
      stroke-opacity: 0.6;
    }

    .nodes circle {
      stroke: #fff;
      stroke-width: 1.5px;
    }''') #create node
    style.appendChild(style_text)
    
    
    background = doc.createElement('rect')
    svg.appendChild(background)
    background.setAttribute('width','100%')
    background.setAttribute('height','100%')
    background.setAttribute('style',"fill:rgb(255,255,255);")
    
    if not text is None:
        for i in range(len(text)):
            g_text = doc.createElement('g')
            g_text.setAttribute('class',"text")
            svg.appendChild(g_text)
            real_text = doc.createElement('text')
            real_text.setAttribute('x',str(50))
            real_text.setAttribute('y',str(20+i*15))
            real_content = doc.createTextNode(text[i])
            real_text.appendChild(real_content)
            g_text.appendChild(real_text)
                             
    
    g_nodes = doc.createElement('g')
    g_nodes.setAttribute('class',"nodes")


    def getx(id):
        if mode == 1:
            x = y[id][0]*float(width)
        else:
            x = y[id][0]
        if x<0:
            x = 0
        if x>width:
            x = width
        return x
    def gety(id):
        if mode == 1:
            y_value = y[id][1]*float(height)
        else:
            y_value = y[id][1]
        if y_value < 0:
            y_value = 0
        if y_value > height:
            y_value = height
        return y_value
    node_mapping = {}
    for node in ori['nodelist']:
        id = node[0]
        if mode == 0:
            cx = node[2]
            cy = node[3]
        else:
            cx = getx(id)
            cy = gety(id)
        if id in node_mapping:
            print("node mapping error" + str(id))
        else:
            node_mapping[id] = [cx,cy]
        node_group = node[1]
        r = node[4]
        fill = node[5]
        circle = doc.createElement('circle')
        circle.setAttribute('stroke',"#fff")
        circle.setAttribute('stroke-width',"1.5px")
        circle.setAttribute('r',str(r))
        circle.setAttribute('fill',fill)
        circle.setAttribute('node_id',str(id))
        circle.setAttribute('node_group',str(node_group))
        circle.setAttribute('cx',str(cx))
        circle.setAttribute('cy',str(cy))
        title = doc.createElement('title')
        title_text = doc.createTextNode(str(id))
        title.appendChild(title_text)
        circle.appendChild(title)
        g_nodes.appendChild(circle)    
    g_links = doc.createElement('g')
    g_links.setAttribute('class',"links")
    for line in ori['linelist']:
        node1 = line[0]
        node2 = line[1]
        if not mode==0:
            x1 = getx(node1)
            y1 = gety(node1)
            x2 = getx(node2)
            y2 = gety(node2)
        else:
            if node1 in node_mapping:
                if node2 in node_mapping:
                    x1 = node_mapping[node1][0]
                    y1 = node_mapping[node1][1]
                    x2 = node_mapping[node2][0]
                    y2 = node_mapping[node2][1]
                else:
                    print("link mapping error : " + str(node2))
            else:
                print("link mapping error : " + str(node1))
        link_ids = str(node1)+"_"+str(node2)
        line = doc.createElement('line')

        line.setAttribute('stroke',"#999")
        line.setAttribute("stroke-opacity","0.6")
        line.setAttribute('stroke-width',"2")
        line.setAttribute('link_ids',link_ids)
        line.setAttribute('x1',str(x1))
        line.setAttribute('y1',str(y1))
        line.setAttribute('x2',str(x2))
        line.setAttribute('y2',str(y2))
        g_links.appendChild(line)
    
    
    svg.appendChild(g_links)
    svg.appendChild(g_nodes)
    #write to file
    f = open(test_folder+name+'.xml','w')
    doc.writexml(f,indent = '\t',newl = '\n', addindent = '\t',encoding='utf-8')
    f.close()
    
    url = test_folder+name+'.xml'
    write_to = test_folder+name+'.png'
    cairosvg.svg2png(url=url, write_to=write_to)