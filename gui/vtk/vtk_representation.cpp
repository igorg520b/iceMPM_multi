#include "vtk_representation.h"
#include "model.h"
#include "parameters_sim.h"
#include "gpu_partition.h"
//#include <omp.h>
#include <algorithm>
#include <iostream>
#include <spdlog/spdlog.h>

icy::VisualRepresentation::VisualRepresentation()
{
    int nLut = sizeof lutArrayTemperatureAdj / sizeof lutArrayTemperatureAdj[0];
    hueLut_temperature->SetNumberOfTableValues(nLut);
    for ( int i=0; i<nLut; i++)
        hueLut_temperature->SetTableValue(i, lutArrayTemperatureAdj[i][0],
                              lutArrayTemperatureAdj[i][1],
                              lutArrayTemperatureAdj[i][2], 1.0);


    nLut = sizeof(lutSouthwest)/sizeof lutSouthwest[0];
    hueLut_Southwest->SetNumberOfTableValues(nLut);
    for ( int i=0; i<nLut; i++)
        hueLut_Southwest->SetTableValue(i, lutSouthwest[i][0],
                                          lutSouthwest[i][1],
                                          lutSouthwest[i][2], 1.0);


    nLut = sizeof lutArrayPastel / sizeof lutArrayPastel[0];
    hueLut_pastel->SetNumberOfTableValues(nLut);
    for ( int i=0; i<nLut; i++)
        hueLut_pastel->SetTableValue(i, lutArrayPastel[i][0],
                              lutArrayPastel[i][1],
                              lutArrayPastel[i][2], 1.0);
    hueLut_pastel->SetTableRange(0,nLut-1);

    nLut = sizeof lutArrayMPMColors / sizeof lutArrayMPMColors[0];
    lutMPM->SetNumberOfTableValues(nLut);
    for ( int i=0; i<nLut; i++)
        lutMPM->SetTableValue(i, lutArrayMPMColors[i][0],
                              lutArrayMPMColors[i][1],
                              lutArrayMPMColors[i][2], 1.0);

    hueLut_four->SetNumberOfColors(5);
    hueLut_four->SetTableValue(0, 0.3, 0.3, 0.3);
    hueLut_four->SetTableValue(1, 1.0, 0, 0);
    hueLut_four->SetTableValue(2, 0, 1.0, 0);
    hueLut_four->SetTableValue(3, 0.2, 0.2, 0.85);
    hueLut_four->SetTableValue(4, 0, 0.5, 0.5);
    hueLut_four->SetTableRange(0,4);


    indenterMapper->SetInputConnection(indenterSource->GetOutputPort());
    actor_indenter->SetMapper(indenterMapper);

    indenterSource->GeneratePolygonOff();
    indenterSource->SetNumberOfSides(50);

    indenterMapper->SetInputConnection(indenterSource->GetOutputPort());
    actor_indenter->SetMapper(indenterMapper);
    actor_indenter->GetProperty()->LightingOff();
    actor_indenter->GetProperty()->EdgeVisibilityOn();
    actor_indenter->GetProperty()->VertexVisibilityOff();
    actor_indenter->GetProperty()->SetColor(0.1,0.1,0.1);
    actor_indenter->GetProperty()->SetEdgeColor(90.0/255.0, 90.0/255.0, 97.0/255.0);
    actor_indenter->GetProperty()->ShadingOff();
    actor_indenter->GetProperty()->SetInterpolationToFlat();
    actor_indenter->PickableOff();
    actor_indenter->GetProperty()->SetLineWidth(3);


    // points
    points_polydata->SetPoints(points);
    points_polydata->GetPointData()->AddArray(visualized_values);
    visualized_values->SetName("visualized_values");
    points_polydata->GetPointData()->SetActiveScalars("visualized_values");

    points_filter->SetInputData(points_polydata);
    points_filter->Update();

    points_mapper->SetInputData(points_filter->GetOutput());
    points_mapper->UseLookupTableScalarRangeOn();
    points_mapper->SetLookupTable(lutMPM);

    actor_points->SetMapper(points_mapper);
    actor_points->GetProperty()->SetPointSize(2);
    actor_points->GetProperty()->SetVertexColor(1,0,0);
    actor_points->GetProperty()->SetColor(0,0,0);
    actor_points->GetProperty()->LightingOff();
    actor_points->GetProperty()->ShadingOff();
    actor_points->GetProperty()->SetInterpolationToFlat();
    actor_points->PickableOff();

    grid_mapper->SetInputData(structuredGrid);
//    grid_mapper->SetLookupTable(hueLut);

    actor_grid->SetMapper(grid_mapper);
    actor_grid->GetProperty()->SetEdgeVisibility(true);
    actor_grid->GetProperty()->SetEdgeColor(0.8,0.8,0.8);
    actor_grid->GetProperty()->LightingOff();
    actor_grid->GetProperty()->ShadingOff();
    actor_grid->GetProperty()->SetInterpolationToFlat();
    actor_grid->PickableOff();
    actor_grid->GetProperty()->SetColor(0.98,0.98,0.98);
//    actor_grid->GetProperty()->SetRepresentationToWireframe();

    // partitions grid
    partitions_grid_mapper->SetInputData(partitionsGrid);
    actor_partitions->SetMapper(partitions_grid_mapper);
    actor_partitions->GetProperty()->SetEdgeVisibility(true);
    actor_partitions->GetProperty()->SetEdgeColor(0.8,0.8,0.8);
    actor_partitions->GetProperty()->LightingOff();
    actor_partitions->GetProperty()->ShadingOff();
    actor_partitions->GetProperty()->SetInterpolationToFlat();
    actor_partitions->PickableOff();
    actor_partitions->GetProperty()->SetColor(0.4,0.4,0.4);
    actor_partitions->GetProperty()->SetRepresentationToWireframe();
    actor_partitions->GetProperty()->SetLineWidth(2);

    // scalar bar
    scalarBar->SetLookupTable(lutMPM);
    scalarBar->SetMaximumWidthInPixels(130);
    scalarBar->SetBarRatio(0.07);
    scalarBar->SetMaximumHeightInPixels(200);
    scalarBar->GetPositionCoordinate()->SetCoordinateSystemToNormalizedDisplay();
    scalarBar->GetPositionCoordinate()->SetValue(0.01,0.015, 0.0);
    scalarBar->SetLabelFormat("%.1e");
    scalarBar->GetLabelTextProperty()->BoldOff();
    scalarBar->GetLabelTextProperty()->ItalicOff();
    scalarBar->GetLabelTextProperty()->ShadowOff();
    scalarBar->GetLabelTextProperty()->SetColor(0.1,0.1,0.1);

    // text
    vtkTextProperty* txtprop = actorText->GetTextProperty();
    txtprop->SetFontFamilyToArial();
    txtprop->BoldOff();
    txtprop->SetFontSize(14);
    txtprop->ShadowOff();
    txtprop->SetColor(0,0,0);
    actorText->SetDisplayPosition(500, 30);

    // water level
    water_points->SetNumberOfPoints(water_level_resolution);
    polyLine->GetPointIds()->SetNumberOfIds(water_level_resolution);
    for(int i=0;i<water_level_resolution;i++) polyLine->GetPointIds()->SetId(i,i);
    water_cells->InsertNextCell(polyLine);
    water_polydata->SetPoints(water_points);
    water_polydata->SetLines(water_cells);
    water_mapper->SetInputData(water_polydata);
    actor_water->SetMapper(water_mapper);
    actor_water->GetProperty()->SetColor(0.1,0.1,0.8);
    actor_water->GetProperty()->SetLineWidth(2.5);
}



void icy::VisualRepresentation::SynchronizeTopology()
{
    model->accessing_point_data.lock();

    spdlog::info("SynchronizeTopology()");

    points->SetNumberOfPoints(model->prms.nPtsTotal);
    visualized_values->SetNumberOfValues(model->prms.nPtsTotal);
    indenterSource->SetRadius(model->prms.IndDiameter/2.f);


    int gx = model->prms.GridXTotal;
    int gy = model->prms.GridY;
    double &h = model->prms.cellsize;
    structuredGrid->SetDimensions(gx, gy, 1);

    grid_points->SetNumberOfPoints(gx*gy);
    for(int idx_y=0; idx_y<gy; idx_y++)
        for(int idx_x=0; idx_x<gx; idx_x++)
        {
            float x = idx_x * h;
            float y = idx_y * h;
            double pt_pos[3] {x, y, -2.0};
            grid_points->SetPoint((vtkIdType)(idx_x+idx_y*gx), pt_pos);
        }
    structuredGrid->SetPoints(grid_points);

    int nPartitions = model->gpu.partitions.size();
    partitionsGrid->SetDimensions(nPartitions+1, 2, 1);
    partitions_grid_points->SetNumberOfPoints((nPartitions+1)*2);
    // partitions grid
    double y1 = -0.5*h;
    double y2 = (model->prms.GridY-0.5)*h;
    for(int i=0;i<nPartitions;i++)
    {
        GPU_Partition &p = model->gpu.partitions[i];
        double x =(p.GridX_offset - 0.5)*h;
        double pt_pos1[3] {x,y1,-1.0};
        partitions_grid_points->SetPoint(i, pt_pos1);
        double pt_pos2[3] {x,y2,-1.0};
        partitions_grid_points->SetPoint(i+nPartitions+1, pt_pos2);
    }

    double x =(model->prms.GridXTotal - 0.5)*h;
    double pt_pos1[3] {x,y1,-0.5};
    partitions_grid_points->SetPoint(nPartitions, pt_pos1);
    double pt_pos2[3] {x,y2,-0.5};
    partitions_grid_points->SetPoint(2*nPartitions + 1, pt_pos2);

    partitionsGrid->SetPoints(partitions_grid_points);

    model->accessing_point_data.unlock();
    SynchronizeValues();
}


void icy::VisualRepresentation::SynchronizeValues()
{
    model->accessing_point_data.lock();

    double sim_time = model->prms.SimulationTime;
    // water level
    for(int i=0;i<water_level_resolution;i++)
    {
        double x = (double)i/water_level_resolution*model->prms.GridXDimension;
        double y = FreeSurfaceElevation(model->prms.SimulationTime, x);
        water_points->SetPoint(i, x, y, 0.);
    }
    water_points->Modified();
//    polyLine->Modified();
//    water_polydata->Modified();


    // spdlog::info("SynchronizeValues() npts {}", model->prms.nPtsTotal);
    double indenter_x = model->prms.indenter_x;
    double indenter_y = model->prms.indenter_y;
    indenterSource->SetCenter(indenter_x, indenter_y, 1);


    unsigned activePtsCount = 0;
    for(int i=0;i<model->gpu.hssoa.size;i++)
    {
        SOAIterator s = model->gpu.hssoa.begin()+i;
        if(s->getDisabledStatus()) continue;
        Eigen::Vector2d pos = s->getPos();
        points->SetPoint((vtkIdType)activePtsCount, pos[0], pos[1], 0);
//        spdlog::info("setting point {}; {} - {}", activePtsCount, pos[0], pos[1]);
        activePtsCount++;
    }
    if(activePtsCount != model->prms.nPtsTotal) throw std::runtime_error("SynchronizeValues() point count mismatch (pos)");
    points->Modified();
    actor_points->GetProperty()->SetPointSize(model->prms.ParticleViewSize);
    points_filter->Update();
    double range = std::pow(10,ranges[VisualizingVariable]);
    double centerVal = 0;


    if(VisualizingVariable == VisOpt::partition)
    {
        scalarBar->VisibilityOn();
        points_mapper->ScalarVisibilityOn();
        points_mapper->SetColorModeToMapScalars();
        points_mapper->UseLookupTableScalarRangeOn();
        points_mapper->SetLookupTable(hueLut_pastel);
        scalarBar->SetLookupTable(hueLut_pastel);

        visualized_values->SetNumberOfValues(model->prms.nPtsTotal);
        activePtsCount = 0;
        for(int i=0;i<model->gpu.hssoa.size;i++)
        {
            SOAIterator s = model->gpu.hssoa.begin()+i;
            if(s->getDisabledStatus()) continue;
            uint8_t partition = s->getPartition();
            bool isCrushed = s->getCrushedStatus();
            if(isCrushed) partition = 41;
            visualized_values->SetValue((vtkIdType)activePtsCount++, (float)partition);
        }
        if(activePtsCount != model->prms.nPtsTotal) throw std::runtime_error("SynchronizeValues() point count mismatch");
        visualized_values->Modified();
    }
    else if(VisualizingVariable == VisOpt::status)
    {
        scalarBar->VisibilityOn();
        points_mapper->ScalarVisibilityOn();
        points_mapper->SetColorModeToMapScalars();
        points_mapper->UseLookupTableScalarRangeOn();
        points_mapper->SetLookupTable(hueLut_four);
        scalarBar->SetLookupTable(hueLut_four);
        visualized_values->SetNumberOfValues(model->prms.nPtsTotal);
        activePtsCount = 0;
        for(int i=0;i<model->gpu.hssoa.size;i++)
        {
            SOAIterator s = model->gpu.hssoa.begin()+i;
            if(s->getDisabledStatus()) continue;
            bool isCrushed = s->getCrushedStatus();
            float value = 0;
            if(isCrushed) value = 1;
            if(s->getLiquidStatus()) value = 3;
            visualized_values->SetValue((vtkIdType)activePtsCount++, (float)value);
        }
        if(activePtsCount != model->prms.nPtsTotal) throw std::runtime_error("SynchronizeValues() point count mismatch");
        visualized_values->Modified();
    }
    else if(VisualizingVariable == VisOpt::Jp_inv)
    {
        scalarBar->VisibilityOn();
        points_mapper->ScalarVisibilityOn();
        points_mapper->SetColorModeToMapScalars();
        points_mapper->UseLookupTableScalarRangeOn();
        points_mapper->SetLookupTable(lutMPM);
        scalarBar->SetLookupTable(lutMPM);
        lutMPM->SetTableRange(centerVal-range, centerVal+range);
        visualized_values->SetNumberOfValues(model->prms.nPtsTotal);
        activePtsCount = 0;
        for(int i=0;i<model->gpu.hssoa.size;i++)
        {
            SOAIterator s = model->gpu.hssoa.begin()+i;
            if(s->getDisabledStatus()) continue;
            double value = s->getValue(icy::SimParams::idx_Jp_inv)-1;
            visualized_values->SetValue((vtkIdType)activePtsCount++, (float)value);
        }
        if(activePtsCount != model->prms.nPtsTotal) throw std::runtime_error("SynchronizeValues() point count mismatch");
        visualized_values->Modified();
    }
    else if(VisualizingVariable == VisOpt::grains)
    {
        scalarBar->VisibilityOn();
        points_mapper->ScalarVisibilityOn();
        points_mapper->SetColorModeToMapScalars();
        points_mapper->UseLookupTableScalarRangeOn();
        points_mapper->SetLookupTable(hueLut_pastel);
        scalarBar->SetLookupTable(hueLut_pastel);

        visualized_values->SetNumberOfValues(model->prms.nPtsTotal);
        activePtsCount = 0;
        for(int i=0;i<model->gpu.hssoa.size;i++)
        {
            SOAIterator s = model->gpu.hssoa.begin()+i;
            if(s->getDisabledStatus()) continue;
            uint16_t grain = s->getGrain()%40;
            bool isCrushed = s->getCrushedStatus();
            if(isCrushed) grain = 41;
            visualized_values->SetValue((vtkIdType)activePtsCount++, (float)grain);
        }
        if(activePtsCount != model->prms.nPtsTotal) throw std::runtime_error("SynchronizeValues() point count mismatch");
        visualized_values->Modified();
    }
    else if(VisualizingVariable == VisOpt::velocity)
    {
        scalarBar->VisibilityOn();
        points_mapper->ScalarVisibilityOn();
        points_mapper->SetColorModeToMapScalars();
        points_mapper->UseLookupTableScalarRangeOn();
        points_mapper->SetLookupTable(hueLut_Southwest);
        scalarBar->SetLookupTable(hueLut_Southwest);
//        hueLut_temperature->SetTableRange(0, 7.0);
        hueLut_Southwest->SetRange(0, 2.0);

        //points_mapper->SetScalarRange(11,-1);
        visualized_values->SetNumberOfValues(model->prms.nPtsTotal);
        activePtsCount = 0;
        for(int i=0;i<model->gpu.hssoa.size;i++)
        {
            SOAIterator s = model->gpu.hssoa.begin()+i;
            if(s->getDisabledStatus()) continue;
            Eigen::Vector2d vel = s->getVelocity();
            float value = (float)vel.norm();
            if(value>=1.9) value=1.9;
            if(!s->getLiquidStatus()) value = 5;
            visualized_values->SetValue((vtkIdType)activePtsCount++, (float)value);
        }
        if(activePtsCount != model->prms.nPtsTotal) throw std::runtime_error("SynchronizeValues() point count mismatch");
        visualized_values->Modified();
    }
    else
    {
        points_mapper->ScalarVisibilityOff();
        scalarBar->VisibilityOff();
    }
    model->accessing_point_data.unlock();

}


void icy::VisualRepresentation::ChangeVisualizationOption(int option)
{
    VisualizingVariable = (VisOpt)option;
    SynchronizeTopology();
}

