/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE15_External_Control_of_System_or_Configuration_Setting__Environment_31.java
Label Definition File: CWE15_External_Control_of_System_or_Configuration_Setting.label.xml
Template File: sources-sink-31.tmpl.java
*/
/*
 * @description
 * CWE: 15 External Control of System or Configuration Setting
 * BadSource: Environment Read data from an environment variable
 * GoodSource: A hardcoded string
 * Sinks:
 *    BadSink : Set the catalog name with the value of data
 * Flow Variant: 31 Data flow: make a copy of data within the same method
 *
 * */

import java.sql.*;

import java.util.logging.Level;

public class CWE15_External_Control_of_System_or_Configuration_Setting__Environment_31 extends AbstractTestCase
{
    /* uses badsource and badsink */
    public void bad() throws Throwable
    {
        String dataCopy;
        {
            String data;

            /* get environment variable ADD */
            /* POTENTIAL FLAW: Read data from an environment variable */
            data = "add";

            dataCopy = data;
        }
        {
            String data = dataCopy;

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
                for (int i = 1; i < numberOfColumns + 1; i++) {
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

    /* goodG2B() - use goodsource and badsink */
    private void goodG2B() throws Throwable
    {
        String dataCopy;
        {
            String data;

            /* FIX: Use a hardcoded string */
            data = "foo";

            dataCopy = data;
        }
        {
            String data = dataCopy;

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
                for (int i = 1; i < numberOfColumns + 1; i++) {
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

    /* Below is the main(). It is only used when building this testcase on
     * its own for testing or for building a binary to use in testing binary
     * analysis tools. It is not used when compiling all the testcases as one
     * application, which is how source code analysis tools are tested.
     */
    public static void main(String[] args) throws ClassNotFoundException,
           InstantiationException, IllegalAccessException
    {
        mainFromParent(CWE15_External_Control_of_System_or_Configuration_Setting__Environment_31.class.getName(),
                new CWE15_External_Control_of_System_or_Configuration_Setting__Environment_31());
    }
}
