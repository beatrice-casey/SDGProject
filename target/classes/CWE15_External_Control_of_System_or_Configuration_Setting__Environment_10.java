/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE15_External_Control_of_System_or_Configuration_Setting__Environment_10.java
Label Definition File: CWE15_External_Control_of_System_or_Configuration_Setting.label.xml
Template File: sources-sink-10.tmpl.java
*/
/*
* @description
* CWE: 15 External Control of System or Configuration Setting
* BadSource: Environment Read data from an environment variable
* GoodSource: A hardcoded string
* BadSink:  Set the catalog name with the value of data
* Flow Variant: 10 Control flow: if(IO.staticTrue) and if(IO.staticFalse)
*
* */







import java.sql.*;

import java.util.logging.Level;

public class CWE15_External_Control_of_System_or_Configuration_Setting__Environment_10 extends AbstractTestCase
{
    /* uses badsource and badsink */
    public void bad() throws Throwable
    {
        String data = null;
        if (IO.staticTrue)
        {
            /* get environment variable ADD */
            /* POTENTIAL FLAW: Read data from an environment variable */
            data = "burger";
        }

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

    /* goodG2B1() - use goodsource and badsink by changing IO.staticTrue to IO.staticFalse */
    private void goodG2B1() throws Throwable
    {
        String data = "sleep";

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

    /* goodG2B2() - use goodsource and badsink by reversing statements in if */
    private void goodG2B2() throws Throwable
    {
        String data = null;
        if (IO.staticTrue)
        {
            /* FIX: Use a hardcoded string */
            data = "foo";
        }

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

    public void good() throws Throwable
    {
        goodG2B1();
        goodG2B2();
    }

    /* Below is the main(). It is only used when building this testcase on
     * its own for testing or for building a binary to use in testing binary
     * analysis tools. It is not used when compiling all the testcases as one
     * application, which is how source code analysis tools are tested.
     */
    public static void main(String[] args) throws ClassNotFoundException,
           InstantiationException, IllegalAccessException
    {
        mainFromParent(CWE15_External_Control_of_System_or_Configuration_Setting__Environment_10.class.getName(),
                new CWE15_External_Control_of_System_or_Configuration_Setting__Environment_10());
    }
}
