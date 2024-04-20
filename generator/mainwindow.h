#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <QVTKOpenGLNativeWidget.h>

#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>
#include <vtkProperty.h>
#include <vtkNew.h>

#include <spdlog/spdlog.h>

#include "vtkrepresentation.h"
#include "grainprocessor.h"


QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void Process();
    VtkRepresentation representation;

private:
    Ui::MainWindow *ui;

    // VTK
    vtkNew<vtkGenericOpenGLRenderWindow> renderWindow;
    QVTKOpenGLNativeWidget *qt_vtk_widget;
    vtkNew<vtkRenderer> renderer;
};
#endif // MAINWINDOW_H
