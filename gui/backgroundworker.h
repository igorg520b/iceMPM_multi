#ifndef BACKGROUNDWORKER_H
#define BACKGROUNDWORKER_H

#include <QObject>
#include <QDebug>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QString>
#include "model.h"

class BackgroundWorker : public QThread
{
    Q_OBJECT
public:
    BackgroundWorker(icy::Model *controller_);
    void Pause();       // cancel current step and pause the worker thread
    void Resume();      // resume the worker thread
    void Finalize();    // exit the worker thread

    bool timeToPause = true;
    bool running = false;
    bool visual_update_requested = false;
    icy::Model *controller;

protected:
    void run() override;

private:
    QMutex mutex;
    QWaitCondition condition;
    bool kill = false; // set "true" and call Resume() to finish run()

Q_SIGNALS:
    void workerPaused();
    void stepCompleted();
};

#endif // BACKGROUNDWORKER_H
