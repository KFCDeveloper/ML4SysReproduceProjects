var MongoClient = require('mongodb').MongoClient;
var url = "mongodb://localhost:27017/";

const args = process.argv.slice(2);
console.log('Arguments: ', args);
if (args.length != 1) {
    return;
}

const tableName = args[0];
const fileName = tableName.concat(".csv");
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
