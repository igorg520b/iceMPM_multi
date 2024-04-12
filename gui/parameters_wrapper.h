#ifndef PARAMETERS_WRAPPER_H
#define PARAMETERS_WRAPPER_H


#include <QObject>
#include <QString>
#include "simulation/parameters_sim.h"
#include <cmath>

// wrapper for SimParams to display/edit them in GUI
class ParamsWrapper : public QObject
{
    Q_OBJECT

    icy::SimParams *prms;

    Q_PROPERTY(double in_TimeStep READ getTimeStep WRITE setTimeStep NOTIFY propertyChanged)
    double getTimeStep() {return prms->InitialTimeStep;}
    void setTimeStep(double val) { prms->InitialTimeStep = val; prms->ComputeHelperVariables();}

    Q_PROPERTY(QString in_TimeStep_ READ getTimeStep_ NOTIFY propertyChanged)
    QString getTimeStep_() {return QString("%1 s").arg(prms->InitialTimeStep,0,'e',1);}

    Q_PROPERTY(double in_SimulationTime READ getSimulationTime WRITE setSimulationTime NOTIFY propertyChanged)
    double getSimulationTime() {return prms->SimulationEndTime;}
    void setSimulationTime(double val) { prms->SimulationEndTime = val; }

    Q_PROPERTY(int in_UpdateEvery READ getUpdateEveryNthStep NOTIFY propertyChanged)
    int getUpdateEveryNthStep() {return prms->UpdateEveryNthStep;}

    Q_PROPERTY(double p_YoungsModulus READ getYoungsModulus WRITE setYoungsModulus NOTIFY propertyChanged)
    double getYoungsModulus() {return prms->YoungsModulus;}
    void setYoungsModulus(double val) { prms->YoungsModulus = (float)val; prms->ComputeLame(); }

    Q_PROPERTY(QString p_YM READ getYM NOTIFY propertyChanged)
    QString getYM() {return QString("%1 Pa").arg(prms->YoungsModulus, 0, 'e', 2);}

    Q_PROPERTY(double p_PoissonsRatio READ getPoissonsRatio WRITE setPoissonsRatio NOTIFY propertyChanged)
    double getPoissonsRatio() {return prms->PoissonsRatio;}
    void setPoissonsRatio(double val) { prms->PoissonsRatio = (float)val; prms->ComputeLame(); }

    Q_PROPERTY(double p_LameLambda READ getLambda NOTIFY propertyChanged)
    double getLambda() {return prms->lambda;}

    Q_PROPERTY(double p_LameMu READ getMu NOTIFY propertyChanged)
    double getMu() {return prms->mu;}

    Q_PROPERTY(double p_LameKappa READ getKappa NOTIFY propertyChanged)
    double getKappa() {return prms->kappa;}


    Q_PROPERTY(double p_ParticleViewSize READ getParticleViewSize WRITE setParticleViewSize NOTIFY propertyChanged)
    double getParticleViewSize() {return prms->ParticleViewSize;}
    void setParticleViewSize(double val) {prms->ParticleViewSize=val;}


    // indenter
    Q_PROPERTY(double IndDiameter READ getIndDiameter NOTIFY propertyChanged)
    double getIndDiameter() {return prms->IndDiameter;}

    Q_PROPERTY(double IndVelocity READ getIndVelocity WRITE setIndVelocity NOTIFY propertyChanged)
    double getIndVelocity() {return prms->IndVelocity;}
    void setIndVelocity(double val) {prms->IndVelocity=val;}

    Q_PROPERTY(double IndDepth READ getIndDepth NOTIFY propertyChanged)
    double getIndDepth() {return prms->IndDepth;}

    // ice block
    Q_PROPERTY(int b_PtActual READ getPointCountActual NOTIFY propertyChanged)
    int getPointCountActual() {return prms->nPtsTotal;}

    Q_PROPERTY(QString b_Grid READ getGridDimensions NOTIFY propertyChanged)
    QString getGridDimensions() {return QString("%1 x %2").arg(prms->GridXTotal).arg(prms->GridY);}

    Q_PROPERTY(double nacc_beta READ getNaccBeta NOTIFY propertyChanged)
    double getNaccBeta() {return prms->NACC_beta;}

    Q_PROPERTY(double nacc_pc READ getNaccPc NOTIFY propertyChanged)
    double getNaccPc() {return (1 - prms->NACC_beta)*prms->IceCompressiveStrength/2.;}

    Q_PROPERTY(double nacc_M READ getNaccM NOTIFY propertyChanged)
    double getNaccM() {return sqrt(prms->NACC_M);}



    // Drucker-Prager
    Q_PROPERTY(double DP_threshold_p READ getDP_threshold_p WRITE setDP_threshold_p NOTIFY propertyChanged)
    double getDP_threshold_p() {return prms->DP_threshold_p;}
    void setDP_threshold_p(double val) {prms->DP_threshold_p = val;}

    Q_PROPERTY(double DP_phi READ getDPPhi WRITE setDPPhi NOTIFY propertyChanged)
    double getDPPhi() {return std::atan(prms->DP_tan_phi)*180/icy::SimParams::pi;}
    void setDPPhi(double val) {prms->DP_tan_phi = tan(val*icy::SimParams::pi/180);}

    Q_PROPERTY(double DP_tan_phi READ getDPTanPhi NOTIFY propertyChanged)
    double getDPTanPhi() {return prms->DP_tan_phi;}


    Q_PROPERTY(double ice_CompressiveStr READ getIce_CompressiveStr WRITE setIce_CompressiveStr NOTIFY propertyChanged)
    double getIce_CompressiveStr() {return prms->IceCompressiveStrength;}
    void setIce_CompressiveStr(double val) {prms->IceCompressiveStrength = val; prms->ComputeCamClayParams2();}

    Q_PROPERTY(double ice_TensileStr READ getIce_TensileStr WRITE setIce_TensileStr NOTIFY propertyChanged)
    double getIce_TensileStr() {return prms->IceTensileStrength;}
    void setIce_TensileStr(double val) {prms->IceTensileStrength = val; prms->ComputeCamClayParams2();}

    Q_PROPERTY(double ice_ShearStr READ getIce_ShearStr WRITE setIce_ShearStr NOTIFY propertyChanged)
    double getIce_ShearStr() {return prms->IceShearStrength;}
    void setIce_ShearStr(double val) {prms->IceShearStrength = val; prms->ComputeCamClayParams2();}


    Q_PROPERTY(int tpb_P2G READ get_tpb_P2G WRITE set_tpb_P2G NOTIFY propertyChanged)
    int get_tpb_P2G() {return prms->tpb_P2G;}
    void set_tpb_P2G(int val) { prms->tpb_P2G = val; }

    Q_PROPERTY(int tpb_Upd READ get_tpb_Upd WRITE set_tpb_Upd NOTIFY propertyChanged)
    int get_tpb_Upd() {return prms->tpb_Upd;}
    void set_tpb_Upd(int val) { prms->tpb_Upd = val; }

    Q_PROPERTY(int tpb_G2P READ get_tpb_G2P WRITE set_tpb_G2P NOTIFY propertyChanged)
    int get_tpb_G2P() {return prms->tpb_G2P;}
    void set_tpb_G2P(int val) { prms->tpb_G2P = val; }

    Q_PROPERTY(double pt_per_cell READ getPtPerCell NOTIFY propertyChanged)
    double getPtPerCell() {return prms->PointsPerCell();}

    Q_PROPERTY(double vc_transfer READ getVectorCapacityTransfer NOTIFY propertyChanged)
    double getVectorCapacityTransfer() {return prms->VectorCapacity_transfer;}

    Q_PROPERTY(double haloSize READ getGridHaloSize NOTIFY propertyChanged)
    double getGridHaloSize() {return prms->GridHaloSize;}


public:
    ParamsWrapper(icy::SimParams *p)
    {
        this->prms = p;
        Reset();
    }

    void Reset()
    {
        // it is possible to change parameters here
    }


Q_SIGNALS:
    void propertyChanged();
};



#endif // PARAMETERS_WRAPPER_H
