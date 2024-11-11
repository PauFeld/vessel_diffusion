import networkx as nx
import torch
import vtk
from modelovae import createNode, numerarNodos, Node, read_tree, deserialize
from vtk import vtkXMLPolyDataReader
#import pymeshlab as pm
import re


use_gpu = False
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

def count_fn(f):
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        return f(*args, **kwargs)
    wrapper.count = 0
    return wrapper

def read_vtk(filename):
	polydata_reader = vtkXMLPolyDataReader()
	polydata_reader.SetFileName(filename)
	polydata_reader.Update()
	polydata = polydata_reader.GetOutput()
	return polydata

def read_obj(file):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(file)
    reader.Update()
    return reader.GetOutput()

def vtpToObj (file):
    
    # Create a renderer and set its background color
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)  # Set background color to white

    # Create a render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # Create a render window interactor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Create a mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(file)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Add the actor to the renderer
    renderer.AddActor(actor)

    # Render the scene
    render_window.Render()

    exporter = vtk.vtkOBJExporter()
    exporter.SetInput(render_window)
    exporter.SetFilePrefix(file.split(".")[0])
    exporter.Write()
    return 


def traversefeatures(root, features):
       
    if root is not None:
        traversefeatures(root.left, features)
        try:
            features.append(root.radius.tolist()[0][3])
        except:
            features.append(root.radius.tolist()[3])
        traversefeatures(root.right, features)
        return features


def findParent(node, val, parent=None):
    if node is None:
        return None

    # If the current node is the required node
    if node.data == val:
        # Return the parent node
        return parent
    else:
        # Recursive calls for the children of the current node
        # Set the current node as the new parent
        left_result = findParent(node.left, val, parent=node)
        right_result = findParent(node.right, val, parent=node)

        # If the node is found in the left subtree, return the result
        if left_result is not None:
            return left_result
        # If the node is found in the right subtree, return the result
        elif right_result is not None:
            return right_result
        # If the node is not found in either subtree, return None
        else:
            return None

def recorrido(root, current, d_bif, l_corte, d2, l_corte2):
    if current is not None:
        recorrido(root, current.right, d_bif, l_corte, d2, l_corte2)
       
        if current.isTwoChild():
            padre = findParent(root, current.data)
            d_bif[current.data] = [padre, current.right, current.left]
            if current.right is not None:
                d2[current.right.data] = current.data
                l_corte2.append(current.right.data)
            if current.left is not None:
                d2[current.left.data] = current.data
                l_corte2.append(current.left.data)
            l_corte.append(current.data)
        if current.isLeaf():
            l_corte.append(current.data)
        recorrido(root, current.left, d_bif, l_corte, d2, l_corte2)

        return 


def recorrido2(root, current):
    if current is not None:      
        print("numero de nodo - numero de hijos:",current.data,  current.childs())
        recorrido2(root, current.right)
        recorrido2(root, current.left)
        return 


def rerootear(grafo):
    grafo = grafo.to_undirected()
    
    aRecorrer = []
    numeroNodoInicial = 0
    distancias = nx.floyd_warshall( grafo )

    parMaximo = (-1, -1)
    maxima = -1

    rad = list(grafo.nodes[numeroNodoInicial]['radio'])
    nodoRaiz = Node( numeroNodoInicial, radius =  rad )

    for vecino in grafo.neighbors( numeroNodoInicial ):
        if vecino != numeroNodoInicial:
            aRecorrer.append( (vecino, numeroNodoInicial,nodoRaiz ) )
            
    while len(aRecorrer) != 0:
        nodoAAgregar, numeroNodoPadre,nodoPadre = aRecorrer.pop(0)
        radius = list(grafo.nodes[nodoAAgregar]['radio'])
   
        nodoActual = Node( nodoAAgregar, radius =  radius)
        nodoPadre.agregarHijo( nodoActual )
        for vecino in grafo.neighbors( nodoAAgregar ):
            if vecino != numeroNodoPadre:
                aRecorrer.append( (vecino, nodoAAgregar,nodoActual) )

 
    return nodoRaiz


@count_fn
def traverse_tree(node, points, radius_array, polyLine, cellarray, d, l_corte, l_corte2, d_id):
    if node is not None:
        
        # Add the current node's point to the vtkPoints
        
        radius = node.radius

        try:
            id = points.InsertNextPoint(radius[0][0], radius[0][1], radius[0][2])
        except:
            id = points.InsertNextPoint(radius[0], radius[1], radius[2])
        d_id[id] = node.data
        try:
            radius_array.InsertNextValue(radius[3]/2)  
        except:
            radius_array.InsertNextValue(radius[0][3]/2)  
      
      
        if node.data in  l_corte: 
            # stop the current line and start a new one   
            if node.data in l_corte2 and d[node.data] is not None:
                polyLine.GetPointIds().InsertNextId(d_id[d[node.data]]) 
            polyLine.GetPointIds().InsertNextId(id) 
            
            if polyLine.GetPointIds().GetNumberOfIds() >1:
                cellarray.InsertNextCell(polyLine)
            polyLine = vtk.vtkPolyLine()
        else:
            # Continue adding points to the current line
            if node.data in l_corte2:
                polyLine.GetPointIds().InsertNextId(d[node.data]) 
            polyLine.GetPointIds().InsertNextId(node.data)

        traverse_tree(node.right, points, radius_array, polyLine, cellarray, d, l_corte, l_corte2, d_id)
        polyLine = vtk.vtkPolyLine()
        traverse_tree(node.left, points, radius_array, polyLine, cellarray, d, l_corte, l_corte2, d_id)
        polyLine = vtk.vtkPolyLine()
        # Recursively traverse the left and right subtrees
        

def tree2centerline(tree, n):    

    points = vtk.vtkPoints()
    radius_array = vtk.vtkFloatArray()
    radius_array.SetName("Radius")
    cellarray = vtk.vtkCellArray()
    polyline = vtk.vtkPolyLine()
    li = []
    tree.traverseInorderChilds(tree, li)
    d = {}
    l_corte = []
    d2 = {}
    l_corte2 = []
    recorrido(tree, tree, d, l_corte, d2, l_corte2)

    d_id = {}
    traverse_tree(tree, points, radius_array, polyline, cellarray, d2, l_corte, l_corte2, d_id)
   
    # Create VTK polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().AddArray(radius_array)

    
    polydata.SetLines(cellarray)
    #print(polydata)
    # Write to VTK file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName("centerlinegenerada.vtp")
    writer.SetInputData(polydata)
    writer.Write()
    #print("polydata", polydata)
    return polydata

def read_tree(filename):
    with open(filename, "r") as f:
        byte = f.read() 
        return byte

def deserialize(data):
    if  not data:
        return 
    nodes = data.split(';')  
    #print("node",nodes[3])
    def post_order(nodes):
                
        if nodes[-1] == '#':
            nodes.pop()
            return None
        node = nodes.pop().split('_')
        try:
            data = int(node[0])
        except:
            numbers = re.findall(r'\d+', node[0])
            # Convert the extracted numbers to integers
            data = [int(num) for num in numbers]
        #radius = float(node[1])
        #print("node", node)
        #breakpoint()
        radius = node[1]
        
        rad = radius.split(",")
        
        rad [0] = rad[0].replace('[','')
        rad [3] = rad[3].replace(']','')
        r = []
        for value in rad:
            r.append(float(value))
        #r =[float(num) for num in radius if num.isdigit()]
        r = torch.tensor(r, device=device)
        #breakpoint()
        root = createNode(data, r)
        root.right = post_order(nodes)
        root.left = post_order(nodes)
        
        return root    
    return post_order(nodes)  


def numerar_nodos(root, count):
    if root is not None:
        numerar_nodos(root.left, count)
        root.data = len(count)
        count.append(1)
        numerar_nodos(root.right, count)
        return 

def predict(latent_size, Grassdecoder, mult):
    '''
    z = torch.randn(1, latent_size)
    generated_images = decode_testing3(z, 100, Grassdecoder, mult)
    '''
    tree = read_tree("profane/0003_treep5.dat")
    generated_images = deserialize(tree)
    c = []
    numerar_nodos(generated_images, c)
    #generated_images = deserialize(read_tree("p10ArteryObjAN1-0_tree.dat"))
    
    count = []
    numerarNodos(generated_images, count)
    
    r_list = []
    r_list = traversefeatures(generated_images, r_list)
    #print("numero de nodos", len(r_list))
    max_radius = max(r_list)
    min_radius = min(r_list)
    centerline = tree2centerline(generated_images, len(r_list))
    

    if abs(max_radius/min_radius)>30:
    #if abs(max_radius/min_radius)>1000000:
        print("ratio muy grande genero de nuevo: ", abs(max_radius/min_radius))
        predict(latent_size, Grassdecoder, mult)
        #raise Exception


    return centerline


def serial2centerline(filename):
    
    
    generated_images = deserialize(read_tree(filename))
    
    count = []
    numerarNodos(generated_images, count)
    
    r_list = []
    r_list = traversefeatures(generated_images, r_list)
    #print("numero de nodos", len(r_list))
    max_radius = max(r_list)
    min_radius = min(r_list)
    centerline = tree2centerline(generated_images, len(r_list))



    return centerline