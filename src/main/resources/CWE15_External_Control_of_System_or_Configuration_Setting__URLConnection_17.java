/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE15_External_Control_of_System_or_Configuration_Setting__URLConnection_17.java
Label Definition File: CWE15_External_Control_of_System_or_Configuration_Setting.label.xml
Template File: sources-sink-17.tmpl.java
*/
/*
* @description
* CWE: 15 External Control of System or Configuration Setting
* BadSource: URLConnection Read data from a web server with URLConnection
* GoodSource: A hardcoded string
* BadSink:  Set the catalog name with the value of data
* Flow Variant: 17 Control flow: for loops
*
* */

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;
import java.net.URL;
import java.net.URLConnection;

import java.util.logging.Level;

import java.sql.*;


public class CWE15_External_Control_of_System_or_Configuration_Setting__URLConnection_17 extends AbstractTestCase
{
    /* uses badsource and badsink */
    public void bad() throws Throwable
    {
        String data;

        data = ""; /* Initialize data */

        /* read input from URLConnection */
        {
            URLConnection urlConnection = (new URL("http://www.example.org/")).openConnection();
            BufferedReader readerBuffered = null;
            InputStreamReader readerInputStream = null;

            try
            {
                readerInputStream = new InputStreamReader(urlConnection.getInputStream(), "UTF-8");
                readerBuffered = new BufferedReader(readerInputStream);

                /* POTENTIAL FLAW: Read data from a web server with URLConnection */
                /* This will be reading the first "line" of the response body,
                 * which could be very long if there are no newlines in the HTML */
                if (readerBuffered.readLine().length() < 30) {
                    data = readerBuffered.readLine();
                }
            }
            catch (IOException exceptIO)
            {
                IO.logger.log(Level.WARNING, "Error with stream reading", exceptIO);
            }
            finally
            {
                /* clean up stream reading objects */
                try
                {
                    if (readerBuffered != null)
                    {
                        readerBuffered.close();
                    }
                }
                catch (IOException exceptIO)
                {
                    IO.logger.log(Level.WARNING, "Error closing BufferedReader", exceptIO);
                }

                try
                {
                    if (readerInputStream != null)
                    {
                        readerInputStream.close();
                    }
                }
                catch (IOException exceptIO)
                {
                    IO.logger.log(Level.WARNING, "Error closing InputStreamReader", exceptIO);
                }
            }
        }

        for (int i = 0; i < 1; i++)
        {
            Connection dbConnection = null;
            try
            {
                dbConnection = IO.getDBConnection();
                /* POTENTIAL FLAW: Set the catalog name with the value of data
                 * allowing a nonexistent catalog name or unauthorized access to a portion of the DB */
                DatabaseMetaData metadata = dbConnection.getMetaData();
                ResultSet rs = metadata.getCatalogs();
                ResultSetMetaData rsMetaData = rs.getMetaData();
                int numberOfColumns = rsMetaData.getColumnCount();

// get the column names; column indexes start from 1
                for (int f = 1; f < numberOfColumns + 1; f++) {
                    String columnName = rsMetaData.getColumnName(i);
                    // Get the name of the column's table name
                    if (data.equals(columnName)) {
                        dbConnection.setCatalog(data);
                    }
                }
            }
            catch (SQLException exceptSql)
            {
                IO.logger.log(Level.WARNING, "Error getting database connection", exceptSql);
            }
            finally
            {
                try
                {
                    if (dbConnection != null)
                    {
                        dbConnection.close();
                    }
                }
                catch (SQLException exceptSql)
                {
                    IO.logger.log(Level.WARNING, "Error closing Connection", exceptSql);
                }
            }
        }
    }

    /* goodG2B() - use goodsource and badsink by reversing the block outside the
     * for statement with the one in the for statement */
    private void goodG2B() throws Throwable
    {
        String data;

        /* FIX: Use a hardcoded string */
        data = "foo";

        for (int i = 0; i < 1; i++)
        {
            Connection dbConnection = null;
            try
            {
                dbConnection = IO.getDBConnection();
                /* POTENTIAL FLAW: Set the catalog name with the value of data
                 * allowing a nonexistent catalog name or unauthorized access to a portion of the DB */
                DatabaseMetaData metadata = dbConnection.getMetaData();
                ResultSet rs = metadata.getCatalogs();
                ResultSetMetaData rsMetaData = rs.getMetaData();
                int numberOfColumns = rsMetaData.getColumnCount();

// get the column names; column indexes start from 1
                for (int f = 1; f < numberOfColumns + 1; f++) {
                    String columnName = rsMetaData.getColumnName(i);
                    // Get the name of the column's table name
                    if (data.equals(columnName)) {
                        dbConnection.setCatalog(data);
                    }
                }
            }
            catch (SQLException exceptSql)
            {
                IO.logger.log(Level.WARNING, "Error getting database connection", exceptSql);
            }
            finally
            {
                try
                {
                    if (dbConnection != null)
                    {
                        dbConnection.close();
                    }
                }
                catch (SQLException exceptSql)
                {
                    IO.logger.log(Level.WARNING, "Error closing Connection", exceptSql);
                }
            }
        }
    }

    public void good() throws Throwable
    {
        goodG2B();
    }

    /* Below is the main(). It is only used when building this testcase on
     * its own for testing or for building a binary to use in testing binary
     * analysis tools. It is not used when compiling all the testcases as one
     * application, which is how source code analysis tools are tested.
     */
    public static void main(String[] args) throws ClassNotFoundException,
           InstantiationException, IllegalAccessException
    {
        mainFromParent(CWE15_External_Control_of_System_or_Configuration_Setting__URLConnection_17.class.getName(),
                new CWE15_External_Control_of_System_or_Configuration_Setting__URLConnection_17());
    }
}
