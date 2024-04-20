#ifndef VTKREPRESENTATION_H
#define VTKREPRESENTATION_H


#include <vtkNew.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCellType.h>
#include <vtkDataSetMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkNamedColors.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkIntArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkDataSetMapper.h>
#include <vtkLookupTable.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyLine.h>
#include <vtkPointData.h>
#include <vtkCellArray.h>

#include <vtkMutableUndirectedGraph.h>
#include <vtkCellData.h>
#include <vtkUnstructuredGridGeometryFilter.h>
#include <vtkIdFilter.h>
#include <vtkCellCenters.h>
#include <vtkGlyph3D.h>
#include <vtkArrowSource.h>

#include <vtkCubeSource.h>

#include <vtkVertexGlyphFilter.h>

#include "grainprocessor.h"

class VtkRepresentation
{
public:
    VtkRepresentation();

    void SynchronizeTopology();

    GrainProcessor *gp;

    vtkNew<vtkUnstructuredGrid> ugrid;
    vtkNew<vtkCellArray> cellArray;
    vtkNew<vtkDataSetMapper> dataSetMapper;
    vtkNew<vtkActor> actor_grains;

    vtkNew<vtkLookupTable> hueLut_pastel;
    vtkNew<vtkDoubleArray> visualized_values;

    vtkNew<vtkPoints> points_mesh;


    // cube
    vtkNew<vtkCubeSource> source_cube;
    vtkNew<vtkDataSetMapper> mapper_cube;
    vtkNew<vtkActor> actor_cube;

    // points
    vtkNew<vtkPoints> points_MPM;
    vtkNew<vtkPolyData> points_polydata;
    vtkNew<vtkPolyDataMapper> points_mapper;
    vtkNew<vtkVertexGlyphFilter> points_filter;
    vtkNew<vtkFloatArray> visualized_values_MPM;
    vtkNew<vtkActor> actor_MPM;


    static constexpr float lutArrayPastel[40][3] = {
        {196/255.0,226/255.0,252/255.0}, // 0
        {136/255.0,119/255.0,187/255.0},
        {190/255.0,125/255.0,183/255.0},
        {243/255.0,150/255.0,168/255.0},
        {248/255.0,187/255.0,133/255.0},
        {156/255.0,215/255.0,125/255.0},
        {198/255.0,209/255.0,143/255.0},
        {129/255.0,203/255.0,178/255.0},
        {114/255.0,167/255.0,219/255.0},
        {224/255.0,116/255.0,129/255.0},
        {215/255.0,201/255.0,226/255.0},  // 10
        {245/255.0,212/255.0,229/255.0},
        {240/255.0,207/255.0,188/255.0},
        {247/255.0,247/255.0,213/255.0},
        {197/255.0,220/255.0,204/255.0},
        {198/255.0,207/255.0,180/255.0},
        {135/255.0,198/255.0,233/255.0},
        {179/255.0,188/255.0,221/255.0},
        {241/255.0,200/255.0,206/255.0},
        {145/255.0,217/255.0,213/255.0},
        {166/255.0,200/255.0,166/255.0},  // 20
        {199/255.0,230/255.0,186/255.0},
        {252/255.0,246/255.0,158/255.0},
        {250/255.0,178/255.0,140/255.0},
        {225/255.0,164/255.0,195/255.0},
        {196/255.0,160/255.0,208/255.0},
        {145/255.0,158/255.0,203/255.0},
        {149/255.0,217/255.0,230/255.0},
        {193/255.0,220/255.0,203/255.0},
        {159/255.0,220/255.0,163/255.0},
        {235/255.0,233/255.0,184/255.0},  // 30
        {237/255.0,176/255.0,145/255.0},
        {231/255.0,187/255.0,212/255.0},
        {209/255.0,183/255.0,222/255.0},
        {228/255.0,144/255.0,159/255.0},
        {147/255.0,185/255.0,222/255.0},  // 35
        {158/255.0,213/255.0,194/255.0},  // 36
        {177/255.0,201/255.0,139/255.0},  // 37
        {165/255.0,222/255.0,141/255.0},  // 38
        {244/255.0,154/255.0,154/255.0}   // 39
    };
};

#endif // VTKREPRESENTATION_H
