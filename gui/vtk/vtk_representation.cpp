#include "vtk_representation.h"
#include "model.h"
#include "parameters_sim.h"
//#include <omp.h>
#include <algorithm>
#include <iostream>
#include <spdlog/spdlog.h>

icy::VisualRepresentation::VisualRepresentation()
{
    int nLut = sizeof lutArrayTemperatureAdj / sizeof lutArrayTemperatureAdj[0];
//    hueLut->SetNumberOfTableValues(nLut);
//    for ( int i=0; i<nLut; i++)
//        hueLut->SetTableValue(i, lutArrayTemperatureAdj[i][0],
//                              lutArrayTemperatureAdj[i][1],
//                              lutArrayTemperatureAdj[i][2], 1.0);

    nLut = sizeof lutArrayPastel / sizeof lutArrayPastel[0];
    hueLut_pastel->SetNumberOfTableValues(nLut);
    for ( int i=0; i<nLut; i++)
        hueLut_pastel->SetTableValue(i, lutArrayPastel[i][0],
                              lutArrayPastel[i][1],
                              lutArrayPastel[i][2], 1.0);
    hueLut_pastel->SetTableRange(0,40);

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
    hueLut_four->SetTableValue(3, 0, 0, 1.0);
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
    actor_grid->GetProperty()->SetColor(0.95,0.95,0.95);

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
}



void icy::VisualRepresentation::SynchronizeTopology()
{
    spdlog::info("SynchronizeTopology()");
    points->SetNumberOfPoints(model->prms.nPts);
    visualized_values->SetNumberOfValues(model->prms.nPts);
    indenterSource->SetRadius(model->prms.IndDiameter/2.f);

    SynchronizeValues();

    int &gx = model->prms.GridX;
    int &gy = model->prms.GridY;
    double &h = model->prms.cellsize;
    structuredGrid->SetDimensions(model->prms.GridX, model->prms.GridY, 1);

    grid_points->SetNumberOfPoints(gx*gy);
    for(int idx_y=0; idx_y<gy; idx_y++)
        for(int idx_x=0; idx_x<gx; idx_x++)
        {
            float x = idx_x * h;
            float y = idx_y * h;
            double pt_pos[3] {x, y, -1.0};
            grid_points->SetPoint((vtkIdType)(idx_x+idx_y*gx), pt_pos);
        }
    structuredGrid->SetPoints(grid_points);
}


void icy::VisualRepresentation::SynchronizeValues()
{
    spdlog::info("SynchronizeValues() npts {}", model->prms.nPts);
    double indenter_x = model->prms.indenter_x;
    double indenter_y = model->prms.indenter_y;
    indenterSource->SetCenter(indenter_x, indenter_y, 1);

    for(int i=0;i<model->prms.nPts;i++)
    {
        Eigen::Vector2d pos = icy::Point::getPos(model->gpu.tmp_transfer_buffer, model->prms.nPtsPitch, i);
        points->SetPoint((vtkIdType)i, pos[0], pos[1], 0);
    }
    points->Modified();

    double centerVal = 0;
    double range = std::pow(10,ranges[VisualizingVariable]);

    actor_points->GetProperty()->SetPointSize(model->prms.ParticleViewSize);
    points_filter->Update();

    if(VisualizingVariable == VisOpt::NACC_case)
    {
        scalarBar->VisibilityOn();
        points_mapper->ScalarVisibilityOn();
        points_mapper->SetColorModeToMapScalars();
        points_mapper->UseLookupTableScalarRangeOn();
        points_mapper->SetLookupTable(hueLut_four);
        scalarBar->SetLookupTable(hueLut_four);
        hueLut->SetTableRange(centerVal-range, centerVal+range);

        for(int i=0;i<model->prms.nPts;i++)
            visualized_values->SetValue((vtkIdType)i,
                                        icy::Point::getCrushedStatus(model->gpu.tmp_transfer_buffer, model->prms.nPtsPitch, i));
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

        visualized_values->SetNumberOfValues(model->prms.nPts);
        for(int i=0;i<model->prms.nPts;i++)
            visualized_values->SetValue((vtkIdType)i,
                                        icy::Point::getJp_inv(model->gpu.tmp_transfer_buffer, model->prms.nPtsPitch, i)-1);
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

        visualized_values->SetNumberOfValues(model->prms.nPts);
        for(int i=0;i<model->prms.nPts;i++)
        {
            int value = 40;
            uint8_t crushed = icy::Point::getCrushedStatus(model->gpu.tmp_transfer_buffer, model->prms.nPtsPitch, i);
            if(!crushed) value = icy::Point::getGrain(model->gpu.tmp_transfer_buffer, model->prms.nPtsPitch, i)%40;
            visualized_values->SetValue((vtkIdType)i, (float)value);
        }
        visualized_values->Modified();
    }
    else if(VisualizingVariable == VisOpt::special_count)
    {
        scalarBar->VisibilityOn();
        points_mapper->ScalarVisibilityOn();
        points_mapper->SetColorModeToMapScalars();
        points_mapper->UseLookupTableScalarRangeOn();
        points_mapper->SetLookupTable(hueLut);
        scalarBar->SetLookupTable(hueLut);
        hueLut->SetTableRange(0,8);

        visualized_values->SetNumberOfValues(model->prms.nPts);
        for(int i=0;i<model->prms.nPts;i++)
        {
            int value = 40;
            char* ptr_intact = (char*)(&model->gpu.tmp_transfer_buffer[model->prms.nPtsPitch*icy::SimParams::idx_utility_data]);
            uint8_t* ptr_special_count = (uint8_t*)(&ptr_intact[model->prms.nPtsPitch*3]);
            value = ptr_special_count[i];
            visualized_values->SetValue((vtkIdType)i, (float)value);
        }
        visualized_values->Modified();
    }
    else
    {
        points_mapper->ScalarVisibilityOff();
//        points_polydata->GetPointData()->RemoveArray(0);
        scalarBar->VisibilityOff();
    }
    /*

*/

    /*
    actor_points->GetProperty()->SetPointSize(model->prms.ParticleViewSize);

    model->hostside_data_update_mutex.lock();
//#pragma omp parallel
    for(int i=0;i<model->points.size();i++)
    {
        const icy::Point &p = model->points[i];
        double x[3] {p.pos[0], p.pos[1], 0};
        points->SetPoint((vtkIdType)i, x);
      }

    double centerVal = 0;
    double range = std::pow(10,ranges[VisualizingVariable]);
    points_mapper->SetLookupTable(lutMPM);
    scalarBar->SetLookupTable(lutMPM);


    if(VisualizingVariable == VisOpt::none)
    {
        for(int i=0;i<model->points.size();i++) visualized_values->SetValue((vtkIdType)i, 0);
    }
    else if(VisualizingVariable == VisOpt::NACC_case)
    {
        for(int i=0;i<model->points.size();i++) visualized_values->SetValue((vtkIdType)i, model->points[i].q);
        points_mapper->SetLookupTable(hueLut_four);
        scalarBar->SetLookupTable(hueLut_four);
    }
    else if(VisualizingVariable == VisOpt::NACC_case_first)
    {
        for(int i=0;i<model->points.size();i++) visualized_values->SetValue((vtkIdType)i, model->points[i].case_when_Jp_first_changes);
        points_mapper->SetLookupTable(hueLut_four);
        scalarBar->SetLookupTable(hueLut_four);
    }
    else if(VisualizingVariable == VisOpt::Jp)
    {
        for(int i=0;i<model->points.size();i++) visualized_values->SetValue((vtkIdType)i, model->points[i].Jp_inv-1);
    }
    else if(VisualizingVariable == VisOpt::Jp_positive)
    {
        for(int i=0;i<model->points.size();i++) visualized_values->SetValue((vtkIdType)i, model->points[i].Jp_inv>1 ? 1. : 0.);
    }
    else if(VisualizingVariable == VisOpt::zeta)
    {
        for(int i=0;i<model->points.size();i++) visualized_values->SetValue((vtkIdType)i, model->points[i].zeta-1);
    }
    else if(VisualizingVariable == VisOpt::p0)
    {
        for(int i=0;i<model->points.size();i++) visualized_values->SetValue((vtkIdType)i, model->points[i].visualize_p0);
        //centerVal = model->prms.IceCompressiveStrength;
        points_mapper->SetLookupTable(hueLut);
        scalarBar->SetLookupTable(hueLut);
    }
    else if(VisualizingVariable == VisOpt::q_limit)
    {
        for(int i=0;i<model->points.size();i++) visualized_values->SetValue((vtkIdType)i, model->points[i].visualize_q_limit);
        points_mapper->SetLookupTable(hueLut);
        scalarBar->SetLookupTable(hueLut);
    }
    else if(VisualizingVariable == VisOpt::p_tr)
    {
        for(int i=0;i<model->points.size();i++) visualized_values->SetValue((vtkIdType)i, model->points[i].visualize_p);
//        centerVal = (model->prms.IceCompressiveStrength-model->prms.IceTensileStrength)/2;
        points_mapper->SetLookupTable(hueLut);
        scalarBar->SetLookupTable(hueLut);
    }
    else if(VisualizingVariable == VisOpt::q_tr)
    {
        for(int i=0;i<model->points.size();i++) visualized_values->SetValue((vtkIdType)i, model->points[i].visualize_q);
        points_mapper->SetLookupTable(hueLut);
        scalarBar->SetLookupTable(hueLut);
    }


//    float minmax[2];
//    visualized_values->GetValueRange(minmax);
    lutMPM->SetTableRange(centerVal-range, centerVal+range);
    hueLut->SetTableRange(centerVal-range, centerVal+range);

    points->Modified();
    visualized_values->Modified();
    points_filter->Update();
    indenterSource->SetCenter(model->prms.indenter_x, model->prms.indenter_y, 1);

*/
}


void icy::VisualRepresentation::ChangeVisualizationOption(int option)
{
    VisualizingVariable = (VisOpt)option;
    SynchronizeTopology();
}

