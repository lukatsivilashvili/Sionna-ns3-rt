#include "ns3/applications-module.h"
#include "ns3/csma-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/internet-module.h"
#include "ns3/lte-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/sionna-connection-handler.h"
#include "ns3/sionna-helper.h"
#include "ns3/tap-bridge-module.h"
#include "ns3/vector.h"

#include <fstream>
#include <cmath>
#include <limits>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("SimpleSionnaLteDlUdp");

/**
 * Minimal, reproducible LTE downlink UDP experiment with an optional Sionna RT
 * channel (ns3-rt). Produces FlowMonitor XML for throughput/loss plots.
 *
 * Usage examples:
 *   ./ns3 run "simple-sionna-example --sionna=0 --distance=50
 * --outPrefix=results/flowmon/baseline_d50"
 *   ./ns3 run "simple-sionna-example --sionna=1 --distance=50
 * --outPrefix=results/flowmon/sionna_d50"
 *
 * Notes:
 * - For --sionna=0 you should NOT run the Python Sionna server.
 * - For --sionna=1 start the Python Sionna server first.
 */

int
main(int argc, char* argv[])
{
    // -----------------------------
    // CLI parameters (stable defaults)
    // -----------------------------
    bool sionna = true;
    std::string serverIp = "127.0.0.1";
    bool localMachine = true;
    bool verbose = false;
    bool logSionnaMetrics = false;

    bool tap = false;
    std::string tapName = "thetap";
    std::string tapMode = "ConfigureLocal";
    bool tapAppTraffic = false;
    int shutdownSionna = -1; // -1 = auto, 0 = keep alive, 1 = shutdown

    double distance = 30.0;         // meters (UE x-position)
    double enbX = 120.0;
    double enbY = -20.0;
    double enbZ = 11.0;
    double ueX = std::numeric_limits<double>::quiet_NaN();
    double ueY = std::numeric_limits<double>::quiet_NaN();
    double ueZ = std::numeric_limits<double>::quiet_NaN();
    double simTime = 5.0;           // seconds
    std::string outPrefix = "flow"; // output file prefix (can include folders)
    bool flowmon = true;

    uint32_t seed = 1;
    uint32_t run = 1;

    // UDP app parameters
    uint16_t dlPort = 9000;
    uint32_t packetSize = 1200;   // bytes
    double offeredRateMbps = 5.0; // Mbps (keep modest for stability)

    CommandLine cmd;
    cmd.AddValue("sionna", "Use Sionna RT channel (1) or baseline ns-3 (0)", sionna);
    cmd.AddValue("server_ip", "Sionna server IP address", serverIp);
    cmd.AddValue("local_machine", "Connect to Sionna server on local machine", localMachine);
    cmd.AddValue("verbose", "Enable verbose logging in Sionna helper", verbose);
    cmd.AddValue("logSionnaMetrics",
                 "Log Sionna pathgain/delay/LOS once at setup (requires Sionna)",
                 logSionnaMetrics);

    cmd.AddValue("tap", "Expose RemoteHost via TapBridge for host injection", tap);
    cmd.AddValue("tapName", "Tap device name (ConfigureLocal/UseLocal modes)", tapName);
    cmd.AddValue("tapMode", "TapBridge mode: ConfigureLocal|UseLocal|UseBridge", tapMode);
    cmd.AddValue("tapAppTraffic",
                 "When tap=1, also run built-in UDP client/server apps",
                 tapAppTraffic);
    cmd.AddValue("shutdownSionna",
                 "Shutdown Sionna server at end: -1 auto (default), 0 keep alive, 1 shutdown",
                 shutdownSionna);

    cmd.AddValue("distance", "UE distance from eNB in meters", distance);
    cmd.AddValue("enbX", "eNB X position", enbX);
    cmd.AddValue("enbY", "eNB Y position", enbY);
    cmd.AddValue("enbZ", "eNB Z position", enbZ);
    cmd.AddValue("ueX", "UE X position (overrides distance if set)", ueX);
    cmd.AddValue("ueY", "UE Y position (overrides default if set)", ueY);
    cmd.AddValue("ueZ", "UE Z position (overrides default if set)", ueZ);
    cmd.AddValue("simTime", "Simulation time in seconds", simTime);
    cmd.AddValue("outPrefix",
                 "Output prefix for files (e.g., results/flowmon/sionna_d30)",
                 outPrefix);
    cmd.AddValue("flowmon", "Enable FlowMonitor XML output", flowmon);

    cmd.AddValue("seed", "RNG seed", seed);
    cmd.AddValue("run", "RNG run number", run);

    cmd.AddValue("packetSize", "UDP packet size in bytes", packetSize);
    cmd.AddValue("rateMbps", "Offered UDP rate in Mbps", offeredRateMbps);

    cmd.Parse(argc, argv);

    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(run);

    const bool shutdownSionnaFinal = (shutdownSionna < 0) ? !tap : (shutdownSionna != 0);

    if (tap)
    {
        GlobalValue::Bind("SimulatorImplementationType",
                          StringValue("ns3::RealtimeSimulatorImpl"));
        GlobalValue::Bind("ChecksumEnabled", BooleanValue(true));
    }

    // -----------------------------
    // Sionna helper configuration (IMPORTANT: set even when sionna=false)
    // -----------------------------
    SionnaHelper& sionnaHelper = SionnaHelper::GetInstance();
    sionnaHelper.SetSionna(sionna);

    if (sionna)
    {
        sionnaHelper.SetServerIp(serverIp);
        sionnaHelper.SetLocalMachine(localMachine);
        sionnaHelper.SetVerbose(verbose);
    }

    // -----------------------------
    // LTE + EPC topology (1 eNB, 1 UE, 1 RemoteHost)
    // -----------------------------
    Ptr<LteHelper> lteHelper = CreateObject<LteHelper>();
    Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
    lteHelper->SetEpcHelper(epcHelper);

    // Baseline channel model (when Sionna is disabled)
    if (!sionna)
    {
        // Common stochastic baseline
        lteHelper->SetAttribute("PathlossModel",
                                StringValue("ns3::LogDistancePropagationLossModel"));
    }

    NodeContainer enbNodes;
    NodeContainer ueNodes;
    enbNodes.Create(1);
    ueNodes.Create(1);

    Ptr<Node> pgw = epcHelper->GetPgwNode();

    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);

    InternetStackHelper internet;
    internet.Install(remoteHostContainer);
    internet.Install(ueNodes);

    // Link between PGW and remote host (CSMA when tap enabled; P2P otherwise)
    NetDeviceContainer internetDevices;
    if (tap)
    {
        CsmaHelper csma;
        csma.SetChannelAttribute("DataRate", DataRateValue(DataRate("10Gb/s")));
        csma.SetChannelAttribute("Delay", TimeValue(MilliSeconds(1)));
        internetDevices = csma.Install(NodeContainer(pgw, remoteHost));
    }
    else
    {
        PointToPointHelper p2p;
        p2p.SetDeviceAttribute("DataRate", DataRateValue(DataRate("10Gb/s")));
        p2p.SetChannelAttribute("Delay", TimeValue(MilliSeconds(1)));
        internetDevices = p2p.Install(pgw, remoteHost);
    }

    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIfaces = ipv4h.Assign(internetDevices);
    Ipv4Address remoteHostAddr = internetIfaces.GetAddress(1);
    Ipv4Address pgwAddr = internetIfaces.GetAddress(0);

    // Remote host routing to UE subnet
    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    uint32_t remoteIfIndex =
        remoteHost->GetObject<Ipv4>()->GetInterfaceForDevice(internetDevices.Get(1));
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"),
                                               Ipv4Mask("255.0.0.0"),
                                               remoteIfIndex);

    // Mobility: static, reproducible
    MobilityHelper mobility;

    // eNB position
    Ptr<ListPositionAllocator> enbPos = CreateObject<ListPositionAllocator>();
    enbPos->Add(Vector(enbX, enbY, enbZ));
    mobility.SetPositionAllocator(enbPos);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(enbNodes);

    // UE position (absolute if provided, else default based on distance)
    const double uePosX = std::isnan(ueX) ? distance : ueX;
    const double uePosY = std::isnan(ueY) ? 0.0 : ueY;
    const double uePosZ = std::isnan(ueZ) ? 1.5 : ueZ;
    const Vector enbPosVec(enbX, enbY, enbZ);
    const Vector uePosVec(uePosX, uePosY, uePosZ);
    const double actualDistance = CalculateDistance(enbPosVec, uePosVec);
    Ptr<ListPositionAllocator> uePos = CreateObject<ListPositionAllocator>();
    uePos->Add(uePosVec);
    mobility.SetPositionAllocator(uePos);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(ueNodes);

    // Install LTE devices
    NetDeviceContainer enbLteDevs = lteHelper->InstallEnbDevice(enbNodes);
    NetDeviceContainer ueLteDevs = lteHelper->InstallUeDevice(ueNodes);

    // Assign IP to UE
    Ipv4InterfaceContainer ueIpIfaces =
        epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueLteDevs));
    Ipv4Address ueAddr = ueIpIfaces.GetAddress(0);

    // Optional one-time Sionna metrics at setup
    double sionnaPathGain = std::numeric_limits<double>::quiet_NaN();
    double sionnaDelay = std::numeric_limits<double>::quiet_NaN();
    std::string sionnaLos = "";
    if (sionna && logSionnaMetrics)
    {
        const std::string enbId = "obj" + std::to_string(enbNodes.Get(0)->GetId() + 1);
        const std::string ueId = "obj" + std::to_string(ueNodes.Get(0)->GetId() + 1);
        const Vector zeroVel(0.0, 0.0, 0.0);

        updateLocationInSionna(enbId, enbPosVec, 0.0, zeroVel);
        updateLocationInSionna(ueId, uePosVec, 0.0, zeroVel);

        sionnaPathGain = getPathGainFromSionna(enbPosVec, uePosVec);
        sionnaDelay = getPropagationDelayFromSionna(enbPosVec, uePosVec);
        sionnaLos = getLOSStatusFromSionna(enbPosVec, uePosVec);

        NS_LOG_UNCOND("SionnaMetrics: pathgain_db=" << sionnaPathGain
                                                    << " delay_s=" << sionnaDelay
                                                    << " los=" << sionnaLos);
    }

    // Attach UE to eNB
    lteHelper->Attach(ueLteDevs.Get(0), enbLteDevs.Get(0));

    // Default route for UE
    Ptr<Node> ue = ueNodes.Get(0);
    Ptr<Ipv4StaticRouting> ueStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(ue->GetObject<Ipv4>());
    ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);

    // -----------------------------
    // TapBridge: expose RemoteHost link to the host OS
    // -----------------------------
    if (tap)
    {
        TapBridgeHelper tapBridge;
        tapBridge.SetAttribute("Mode", StringValue(tapMode));
        tapBridge.SetAttribute("DeviceName", StringValue(tapName));
        tapBridge.Install(remoteHost, internetDevices.Get(1));

        NS_LOG_UNCOND("TapBridge enabled. Host should see " << tapName << " with IP "
                                                            << remoteHostAddr);
        NS_LOG_UNCOND("Host route: sudo ip route add 7.0.0.0/8 via " << pgwAddr
                                                                    << " dev " << tapName);
        NS_LOG_UNCOND("Try: ping " << ueAddr);
    }

    // -----------------------------
    // Applications: UDP DL RemoteHost -> UE
    // -----------------------------
    ApplicationContainer serverApps;
    ApplicationContainer clientApps;
    const bool enableApps = (!tap) || tapAppTraffic;

    if (enableApps)
    {
        // Server on UE
        UdpServerHelper dlServer(dlPort);
        serverApps = dlServer.Install(ueNodes.Get(0));
        serverApps.Start(Seconds(0.1));
        serverApps.Stop(Seconds(simTime));

        // Client on RemoteHost
        UdpClientHelper dlClient(ueAddr, dlPort);
        dlClient.SetAttribute("Interval", TimeValue(MicroSeconds(200))); // 5 kpps
        dlClient.SetAttribute("PacketSize", UintegerValue(1200));        // ~48 Mbps
        dlClient.SetAttribute("MaxPackets", UintegerValue(0));

        clientApps = dlClient.Install(remoteHost);
        clientApps.Start(Seconds(0.2));
        clientApps.Stop(Seconds(simTime));
    }

    // Optional traces (can be noisy; keep off by default)
    if (verbose)
    {
        lteHelper->EnablePhyTraces();
        lteHelper->EnableMacTraces();
        lteHelper->EnableRlcTraces();
    }

    // -----------------------------
    // FlowMonitor output (optional)
    // -----------------------------
    Ptr<FlowMonitor> monitor;
    std::string flowmonFile;
    if (flowmon)
    {
        FlowMonitorHelper flowHelper;
        monitor = flowHelper.InstallAll();
    }

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    if (flowmon && monitor)
    {
        monitor->CheckForLostPackets();
        flowmonFile = outPrefix + "_flowmon.xml";
        monitor->SerializeToXmlFile(flowmonFile, true, true);
    }

    // Lightweight run summary CSV (useful in reports/scripts)
    // throughput computed from UE received bytes reported by UdpServer
    Ptr<UdpServer> udpServer =
        (enableApps && serverApps.GetN() > 0) ? DynamicCast<UdpServer>(serverApps.Get(0))
                                              : nullptr;

    std::ofstream summary(outPrefix + "_summary.csv", std::ios::out);
    summary << "distance_param_m,distance_actual_m,model,simTime_s,offeredRate_mbps,packetSize_bytes,rx_packets,"
               "sionna_pathgain_db,sionna_delay_s,sionna_los\n";
    summary << distance << "," << actualDistance << "," << (sionna ? "sionna" : "baseline") << ","
            << simTime << "," << offeredRateMbps << "," << packetSize << ","
            << (udpServer ? udpServer->GetReceived() : 0) << ","
            << sionnaPathGain << "," << sionnaDelay << "," << sionnaLos << "\n";
    summary.close();

    Simulator::Destroy();

    // Close Sionna server only if it was used (optional)
    if (sionna && shutdownSionnaFinal)
    {
        sionnaHelper.ShutdownSionna();
    }

    if (flowmon)
    {
        NS_LOG_UNCOND("Wrote FlowMonitor: " << flowmonFile);
    }
    NS_LOG_UNCOND("Wrote Summary: " << outPrefix + "_summary.csv");
    NS_LOG_UNCOND("RemoteHost=" << remoteHostAddr << " UE=" << ueAddr
                                << " distance_param=" << distance << "m"
                                << " distance_actual=" << actualDistance << "m"
                                << " mode=" << (sionna ? "sionna" : "baseline"));

    return 0;
}
