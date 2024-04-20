#include "mainwindow.h"
#include "./ui_mainwindow.h"

#include <filesystem>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // VTK
    qt_vtk_widget = new QVTKOpenGLNativeWidget();
    qt_vtk_widget->setRenderWindow(renderWindow);
    renderer->SetBackground(1.0,1.0,.9);
    renderWindow->AddRenderer(renderer);
    setCentralWidget(qt_vtk_widget);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::Process()
{
    //    renderer->AddActor(representation.actor_grains);
    renderer->AddActor(representation.actor_cube);
    renderer->AddActor(representation.actor_MPM);
    representation.SynchronizeTopology();
}
