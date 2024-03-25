var MongoClient = require('mongodb').MongoClient;

async function main(params) {
  return await handler(payload);
}

// Monitoring function wrapping application payload code
async function handler(payload) {
  var dbName = "serverless-metrics"
  var tableName = "sleep_matmul";

  const child_process = require("child_process");
  const v8 = require("v8");
  const {performance, PerformanceObserver, monitorEventLoopDelay } = require("perf_hooks");
  const [beforeBytesRx, beforePkgsRx, beforeBytesTx, beforePkgsTx] =
    child_process.execSync("cat /proc/net/dev | grep vinternal_1| awk '{print $2,$3,$10,$11}'").toString().split(" ");
  const startUsage = process.cpuUsage();
  const beforeResourceUsage = process.resourceUsage();
  const wrapped = performance.timerify(payload);
  const h = monitorEventLoopDelay();
  h.enable();

  const durationStart = process.hrtime();

  await wrapped();
  h.disable();

  const durationDiff = process.hrtime(durationStart);
  // Duration in ms
  const duration = (durationDiff[0] * 1e9 + durationDiff[1]) / 1e6

  // Process CPU Diff
  const cpuUsageDiff = process.cpuUsage(startUsage);
  // Process Resources
  const afterResourceUsage = process.resourceUsage();

  // Memory
  const heapCodeStats = v8.getHeapCodeStatistics();
  const heapStats = v8.getHeapStatistics();
  const heapInfo = process.memoryUsage();

  // Network
  const [afterBytesRx, afterPkgsRx, afterBytesTx, afterPkgsTx] =
    child_process.execSync("cat /proc/net/dev | grep vinternal_1| awk '{print $2,$3,$10,$11}'").toString().split(" ");

  var url = "mongodb://meta:password@172.17.0.1:27017/serverless-metrics";
  const { v4: uuidv4 } = require('uuid');

  var item = {
    "id": uuidv4(),
    "cpu": 1024,
    "memory": 1024,
    "duration": `${duration}`,
    "maxRss": `${afterResourceUsage.maxRSS - beforeResourceUsage.maxRSS}`,
    "fsRead": `${afterResourceUsage.fsRead - beforeResourceUsage.fsRead}`,
    "fsWrite": `${afterResourceUsage.fsWrite - beforeResourceUsage.fsWrite}`,
    "vContextSwitches": `${afterResourceUsage.voluntaryContextSwitches - beforeResourceUsage.voluntaryContextSwitches}`,
    "ivContextSwitches": `${afterResourceUsage.involuntaryContextSwitches - beforeResourceUsage.involuntaryContextSwitches}`,
    "userDiff": `${cpuUsageDiff.user}`,
    "sysDiff": `${cpuUsageDiff.system}`,
    "rss": `${heapInfo.rss}`,
    "heapTotal": `${heapInfo.heapTotal}`,
    "heapUsed": `${heapInfo.heapUsed}`,
    "external": `${heapInfo.external}`,
    "elMin": `${h.min}`,
    "elMax": `${h.max}`,
    "elMean": `${isNaN(h.mean)? 0 : h.mean}`,
    "elStd": `${isNaN(h.stddev)? 0 : h.stddev}`,
    "bytecodeMetadataSize": `${heapCodeStats.bytecode_and_metadata_size}`,
    "heapPhysical": `${heapStats.total_physical_size}`,
    "heapAvailable": `${heapStats.total_available_size}`,
    "heapLimit": `${heapStats.heap_size_limit}`,
    "mallocMem": `${heapStats.malloced_memory}`,
    "netByRx": `${isNaN(afterBytesRx - beforeBytesRx)? 0 : afterBytesRx - beforeBytesRx}`,
    "netPkgRx": `${isNaN(afterPkgsRx - beforePkgsRx)? 0 : afterPkgsRx - beforePkgsRx}`,
    "netByTx": `${isNaN(afterBytesTx - beforeBytesTx)? 0 : afterBytesTx - beforeBytesTx}`,
    "netPkgTx": `${isNaN(afterPkgsTx - beforePkgsTx)? 0 : afterPkgsTx - beforePkgsTx}`,
  };
  console.log('Duration: %d', duration);

  // Write to the MongoDB
  MongoClient.connect(url, function(err, db) {
    if (err) throw err;
    var dbo = db.db(dbName);
    dbo.collection(tableName).insertOne(item, function(err, res) {
      if (err) throw err;
      console.log("1 record inserted to %s database %s table", dbName, tableName);
      db.close();
    });
  });

  return {
    'statusCode': 200
  };
};


const func0 = async () => {
  const sleepDuration = ~~(Math.random() * 5) + 1;
  return new Promise(resolve => {
    setTimeout(resolve, sleepDuration * 100);
  });
}

const func1 = async () => {
  const mathjs = require("mathjs");
  const dimensions = [25, 50, 75, 100, 125];
  const roll = ~~(Math.random() * 5);
  const matrix1 = mathjs.random([dimensions[roll], dimensions[roll]]);
  const matrix2 = mathjs.random([dimensions[roll], dimensions[roll]]);
  mathjs.multiply(matrix1, matrix2);
}


// Application code
async function payload(event) {

  await func0();
  await func1();
}


main({});
