#include "vtkoffscreen.h"
#include "model.h"


icy::VTKOffscreen::VTKOffscreen()
{
    int nLut = sizeof lutArrayMPMColors / sizeof lutArrayMPMColors[0];
    lutMPM->SetNumberOfTableValues(nLut);
    for ( int i=0; i<nLut; i++)
        lutMPM->SetTableValue(i, lutArrayMPMColors[i][0],
                              lutArrayMPMColors[i][1],
                              lutArrayMPMColors[i][2], 1.0);

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


    points_polydata->SetPoints(points);
    points_polydata->GetPointData()->AddArray(visualized_values);

    points_filter->SetInputData(points_polydata);
    points_filter->Update();

    points_mapper->SetInputData(points_filter->GetOutput());
    points_mapper->UseLookupTableScalarRangeOn();
    points_mapper->SetLookupTable(lutMPM);

    visualized_values->SetName("visualized_values");

    actor_points->SetMapper(points_mapper);
    actor_points->GetProperty()->SetPointSize(2);
    actor_points->GetProperty()->SetVertexColor(1,0,0);
    actor_points->GetProperty()->SetColor(0,0,0);
    actor_points->GetProperty()->LightingOff();
    actor_points->GetProperty()->ShadingOff();
    actor_points->GetProperty()->SetInterpolationToFlat();
    actor_points->PickableOff();


    grid_mapper->SetInputData(structuredGrid);

//    mapper_structuredGrid->SetScalarRange(0, dataSize - 1);
//    mapper_structuredGrid->ScalarVisibilityOn();

    actor_grid->SetMapper(grid_mapper);
    actor_grid->GetProperty()->SetEdgeVisibility(true);
    actor_grid->GetProperty()->SetEdgeColor(0.8,0.8,0.8);
    actor_grid->GetProperty()->LightingOff();
    actor_grid->GetProperty()->ShadingOff();
    actor_grid->GetProperty()->SetInterpolationToFlat();
    actor_grid->PickableOff();
    actor_grid->GetProperty()->SetColor(0.95,0.95,0.95);


    // set up offscreen render
    graphics_factory->SetOffScreenOnlyMode(1);
    graphics_factory->SetUseMesaClasses(1);
    renderWindow->SetOffScreenRendering(1);
    renderWindow->AddRenderer(renderer);
    renderWindow->SetSize(1920,1080);

    renderer->SetBackground(1.0,1.0,1.0);

    renderer->AddActor(actor_grid);
    renderer->AddActor(actor_points);
    renderer->AddActor(actor_indenter);

    windowToImageFilter->SetInput(renderWindow);
    writer->SetInputConnection(windowToImageFilter->GetOutputPort());

    // testing
    sphereMapper->SetInputConnection(sphereSource->GetOutputPort());
    sphereActor->SetMapper(sphereMapper);
//    renderer->AddActor(sphereActor);

    vtkCamera* camera = renderer->GetActiveCamera();
    renderer->ResetCamera();
    camera->ParallelProjectionOn();
    camera->SetClippingRange(1e-1,1e3);
    camera->SetFocalPoint(1.53819 , 0.759985 , 0);
    camera->SetPosition(1.53819 , 0.759985 , 50);
    camera->SetViewUp(0.0, 1.0, 0.0);
    camera->SetParallelScale(1.07824);
    camera->Modified();

}

void icy::VTKOffscreen::SaveScreenshot(std::string fileName)
{
    renderWindow->Render();
    windowToImageFilter->Update();
    windowToImageFilter->Modified();
    writer->Modified();
    writer->SetFileName(fileName.c_str());
    writer->Write();
}

void icy::VTKOffscreen::SaveVTK(std::string fileName)
{
    pdwriter->SetFileName(fileName.c_str());
    pdwriter->SetInputData(points_polydata);
    pdwriter->SetFileTypeToBinary();
    pdwriter->Write();
}



void icy::VTKOffscreen::SynchronizeTopology()
{
    points->SetNumberOfPoints(model->points.size());
    visualized_values->SetNumberOfValues(model->points.size());
    points_polydata->GetPointData()->SetActiveScalars("visualized_values");
    points_mapper->ScalarVisibilityOn();
    points_mapper->SetColorModeToMapScalars();
    float eps = 1.e-3;
    lutMPM->SetTableRange(1.f-eps, 1.f+eps);

    SynchronizeValues();

    // structured grid
    int &gx = model->prms.GridX;
    int &gy = model->prms.GridY;
    real &h = model->prms.cellsize;
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

    // indenter
    indenterSource->SetRadius(model->prms.IndDiameter/2.f);
}


void icy::VTKOffscreen::SynchronizeValues()
{
    actor_points->GetProperty()->SetPointSize(model->prms.ParticleViewSize);

    model->hostside_data_update_mutex.lock();
//#pragma omp parallel
    for(int i=0;i<model->points.size();i++)
    {
        const icy::Point &p = model->points[i];
        double x[3] {p.pos[0], p.pos[1], 0};
        points->SetPoint((vtkIdType)i, x);
        visualized_values->SetValue((vtkIdType)i, p.NACC_alpha_p);
    }

    model->hostside_data_update_mutex.unlock();

    float minmax[2];
    visualized_values->GetValueRange(minmax);
    const float &alpha0 = model->prms.NACC_alpha;
//    float epsilon = std::max(abs(minmax[0]-alpha0),abs(minmax[1]-alpha0));
//    if(epsilon == 0) epsilon = 1e-6;
    float epsilon = 0.02;
    lutMPM->SetTableRange(alpha0-epsilon/2, alpha0+epsilon/2);
//    std::cout << "vis epsilon " << std::scientific << epsilon << '\n';
    points->Modified();
    visualized_values->Modified();
    points_filter->Update();

    indenterSource->SetCenter(model->indenter_x, model->indenter_y, 1);
}




