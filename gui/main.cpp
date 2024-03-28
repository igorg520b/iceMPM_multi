#include "mainwindow.h"
#include <QApplication>
#include <QSurfaceFormat>
#include <QCommandLineParser>
#include <iostream>


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QApplication::setApplicationName("iceMPM");
    QApplication::setApplicationVersion("1.1");

    QCommandLineParser parser;
    parser.setApplicationDescription("MPM simulation of ice with GUI");
    parser.addHelpOption();
    parser.addVersionOption();
    parser.addPositionalArgument("parameters", QCoreApplication::translate("main", "JSON parameter file"));

    parser.process(a);

    const QStringList args = parser.positionalArguments();
    MainWindow w;
    /*
    if(args.size() == 1)
    {
        QString parameters_file = args[0];
        std::string dummy;
        w.model.prms.ParseFile(parameters_file.toStdString(), dummy);
    }
    */

    w.resize(1400,900);
//    w.show();
    w.showMaximized();
    return a.exec();
}
