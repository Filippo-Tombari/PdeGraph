# Nicola R. Franco, PhD, MOX, Politecnico di Milano
# 28 September 2022

import dolfin
import numpy
import gmsh
import os
from IPython.display import clear_output

class Domain(object):
    '''Base class for all domains'''
    def __init__(self, main, other, operation = None):
        """Combines two domains via the specified operation."""
        self.a, self.b, self.op = main, other, operation
        self.index = 0
        self.dim = max(main.dim, other.dim)

    def script(self, index = 1):
        """Writes a gmsh script describing the domain."""
        res, j = self.a.script(index)
        res0, j0 = self.b.script(j)
        self.index = j0
        if(self.op == "u"):
            res0 += "BooleanUnion{%s{%d};}{%s{%d};}\n" % (self.a.entity(), self.a.index, self.b.entity(), self.b.index)
        elif(self.op == "i"):
            res0 += "BooleanIntersection{%s{%d};}{%s{%d};}\n" % (self.a.entity(), self.a.index, self.b.entity(), self.b.index)
        elif(self.op == "d"):
            res0 += "BooleanDifference{%s{%d};}{%s{%d};}\n" % (self.a.entity(), self.a.index, self.b.entity(), self.b.index)
        return res+res0, j0+1

    def __add__(self, other):
        return Domain(self, other, "u")

    def __sub__(self, other):
        return Domain(self, other, "d")

    def __mul__(self, other):
        return Domain(self, other, "i")

    def entity(self):
        if self.dim==2:
            return "Surface"
        elif self.dim==3:
            return "Volume"

class Rectangle(Domain):
    '''A rectangle domain.'''
    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1
        self.index = 0
        self.dim = 2

    def script(self, index = 1):
        self.index = index
        return 'Rectangle(%d) = {%f, %f, 0.0, %f, %f};\n' % (index,self.p0[0],self.p0[1],self.p1[0]-self.p0[0],
                                                             self.p1[1]-self.p0[1]), index+1

class Box(Domain):
    '''A box domain.'''
    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1
        self.index = 0
        self.dim = 3

    def script(self, index = 1):
        self.index = index
        return 'Box(%d) = {%f, %f, %f, %f, %f, %f};\n' % (index,self.p0[0],self.p0[1],self.p0[2],
                                                          self.p1[0]-self.p0[0],self.p1[1]-self.p0[1],self.p1[2]-self.p0[2]), index+1

class Circle(Domain):
    '''A circle domain.'''
    def __init__(self, p, r = 1):
        self.p = p
        self.r = r
        self.index = 0
        self.dim = 2

    def script(self, index = 1):
        self.index = index
        return 'Disk(%d) = {%f, %f, 0.0, %f};\n' % (index,self.p[0], self.p[1], self.r), index+1

class Polygon(Domain):
      '''A polygon domain.'''
      def __init__(self, points):
            self.p = points
            if(numpy.linalg.norm(numpy.array(points[0])- numpy.array(points[-1]))>1e-15):
                raise RuntimeError("First and last point should coincide.")
            self.index = 0
            self.dim = 2

      def script(self, index = 1):
          res = ""
          self.index = index
          n = len(self.p)-1
          for p in self.p[:-1]:
              res += "Point(%d) = {%f, %f, 0.0};\n" % (self.index,p[0],p[1])
              self.index += 1
          base = self.index
          for i in range(n-1):
              res += "Line(%d) = {%d, %d};\n" % (self.index,base-n+i,base-n+1+i)
              self.index += 1
          res += "Line(%d) = {%d, %d};\n" % (self.index,base-1,base-n)
          self.index += 1
          res += "Line Loop(%d) = {" % self.index
          for i in range(n):
              res += "%d, " % (self.index-n+i)
          res = res[:-2] + "};\n"
          self.index += 1
          res += "Plane Surface(%d) = {%d};\n" % (self.index, self.index-1)
          return res, self.index+1


def mesh(domain, stepsize, name,structured = False):
    """Meshes a domain with gmsh
    :param domain: the domain to mesh (a Domain object)
    :param stepsize: the stepsize to use ( float)
    :param name: the name of the mesh file (string)
    :param structured: if True, the mesh will be structured (boolean)
    :return: the mesh (Mesh object)
    """
    if(structured and domain.dim!=2):
        raise RuntimeError("Structured meshes are only available for 2D geometries.")
    code = 'SetFactory("OpenCASCADE");\nMesh.CharacteristicLengthMin = %f;\nMesh.CharacteristicLengthMax = %f;\n' % (stepsize, stepsize)
    code += domain.script()[0]
    extra = "\nTransfinite %s {%d};" %  (domain.entity(), domain.index) if structured else ""
    code += '\nPhysical %s(%d) = {%d};%s\nMesh.MshFileVersion = 2.0;' % (domain.entity(), domain.index+1, domain.index, extra)

    #idf = numpy.random.randint(100000)
    print(code, file = open("%s.geo" %name , 'w'))
    os.system("gmsh -%d %s.geo" % (domain.dim, name))
    clear_output(wait = True)
    os.system("dolfin-convert %s.msh %s.xml" % (name,name))
    clear_output(wait = True)
    mesh = dolfin.cpp.mesh.Mesh("%s.xml" %name)
    os.remove("%s.msh" % name)
    #os.remove("%d.xml" % idf)
    try:
        os.remove("%s_physical_region.xml" % name)
    except:
        None
    os.remove("%s.geo" % name)
    return mesh
