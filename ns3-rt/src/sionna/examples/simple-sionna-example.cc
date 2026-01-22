#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/internet-module.h"
#include "ns3/lte-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/sionna-helper.h"

#include <fstream>

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

    double distance = 30.0;         // meters (UE x-position)
    double simTime = 5.0;           // seconds
    std::string outPrefix = "flow"; // output file prefix (can include folders)

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

    cmd.AddValue("distance", "UE distance from eNB in meters", distance);
    cmd.AddValue("simTime", "Simulation time in seconds", simTime);
    cmd.AddValue("outPrefix",
                 "Output prefix for files (e.g., results/flowmon/sionna_d30)",
                 outPrefix);

    cmd.AddValue("seed", "RNG seed", seed);
    cmd.AddValue("run", "RNG run number", run);

    cmd.AddValue("packetSize", "UDP packet size in bytes", packetSize);
    cmd.AddValue("rateMbps", "Offered UDP rate in Mbps", offeredRateMbps);

    cmd.Parse(argc, argv);

    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(run);

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

    // P2P link between PGW and remote host
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", DataRateValue(DataRate("10Gb/s")));
    p2p.SetChannelAttribute("Delay", TimeValue(MilliSeconds(1)));
    NetDeviceContainer internetDevices = p2p.Install(pgw, remoteHost);

    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIfaces = ipv4h.Assign(internetDevices);
    Ipv4Address remoteHostAddr = internetIfaces.GetAddress(1);

    // Remote host routing to UE subnet
    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    // Mobility: static, reproducible
    MobilityHelper mobility;

    // eNB at (0,0,10)
    Ptr<ListPositionAllocator> enbPos = CreateObject<ListPositionAllocator>();
    enbPos->Add(Vector(120.0, -20.0, 11.0));
    mobility.SetPositionAllocator(enbPos);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(enbNodes);

    // UE at (distance,0,1.5)
    Ptr<ListPositionAllocator> uePos = CreateObject<ListPositionAllocator>();
    uePos->Add(Vector(distance, 0.0, 1.5));
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

    // Attach UE to eNB
    lteHelper->Attach(ueLteDevs.Get(0), enbLteDevs.Get(0));

    // Default route for UE
    Ptr<Node> ue = ueNodes.Get(0);
    Ptr<Ipv4StaticRouting> ueStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(ue->GetObject<Ipv4>());
    ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);

    // -----------------------------
    // Applications: UDP DL RemoteHost -> UE
    // -----------------------------
    // Server on UE
    UdpServerHelper dlServer(dlPort);
    ApplicationContainer serverApps = dlServer.Install(ueNodes.Get(0));
    serverApps.Start(Seconds(0.1));
    serverApps.Stop(Seconds(simTime));

    // Client on RemoteHost
    UdpClientHelper dlClient(ueAddr, dlPort);
    dlClient.SetAttribute("Interval", TimeValue(MicroSeconds(200))); // 5 kpps
    dlClient.SetAttribute("PacketSize", UintegerValue(1200));        // ~48 Mbps
    dlClient.SetAttribute("MaxPackets", UintegerValue(0));

    ApplicationContainer clientApps = dlClient.Install(remoteHost);
    clientApps.Start(Seconds(0.2));
    clientApps.Stop(Seconds(simTime));

    // Optional traces (can be noisy; keep off by default)
    if (verbose)
    {
        lteHelper->EnablePhyTraces();
        lteHelper->EnableMacTraces();
        lteHelper->EnableRlcTraces();
    }

    // -----------------------------
    // FlowMonitor output
    // -----------------------------
    FlowMonitorHelper flowHelper;
    Ptr<FlowMonitor> monitor = flowHelper.InstallAll();

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    monitor->CheckForLostPackets();

    const std::string flowmonFile = outPrefix + "_flowmon.xml";
    monitor->SerializeToXmlFile(flowmonFile, true, true);

    // Lightweight run summary CSV (useful in reports/scripts)
    // throughput computed from UE received bytes reported by UdpServer
    Ptr<UdpServer> udpServer = DynamicCast<UdpServer>(serverApps.Get(0));

    std::ofstream summary(outPrefix + "_summary.csv", std::ios::out);
    summary << "distance_m,model,simTime_s,offeredRate_mbps,packetSize_bytes,rx_packets\n";
    summary << distance << "," << (sionna ? "sionna" : "baseline") << "," << simTime << ","
            << offeredRateMbps << "," << packetSize << ","
            << (udpServer ? udpServer->GetReceived() : 0) << "\n";
    summary.close();

    Simulator::Destroy();

    // Close Sionna server only if it was used
    if (sionna)
    {
        sionnaHelper.ShutdownSionna();
    }

    NS_LOG_UNCOND("Wrote FlowMonitor: " << flowmonFile);
    NS_LOG_UNCOND("Wrote Summary: " << outPrefix + "_summary.csv");
    NS_LOG_UNCOND("RemoteHost=" << remoteHostAddr << " UE=" << ueAddr << " distance=" << distance
                                << "m"
                                << " mode=" << (sionna ? "sionna" : "baseline"));

    return 0;
}
