// var MongoClient = require('mongodb').MongoClient;
// var url = "mongodb://localhost:27017/";

var MongoClient = require('mongodb').MongoClient;
var url = "mongodb://meta:password@localhost:27017/serverless-metrics";

const args = process.argv.slice(2);
console.log('Arguments: ', args);
if (args.length != 1) {
    return;
}

const tableName = args[0];
const dbName = "serverless-metrics";

MongoClient.connect(url, function(err, db) {
  if (err) throw err;
  var dbo = db.db(dbName);
  dbo.collection(tableName).drop(function(err, delOK) {
    if (err) throw err;
    if (delOK) console.log("Collection %s deleted", tableName);
    db.close();
  });
});
