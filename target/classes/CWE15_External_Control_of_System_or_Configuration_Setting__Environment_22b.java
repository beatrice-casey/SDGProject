/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE15_External_Control_of_System_or_Configuration_Setting__Environment_22b.java
Label Definition File: CWE15_External_Control_of_System_or_Configuration_Setting.label.xml
Template File: sources-sink-22b.tmpl.java
*/
/*
 * @description
 * CWE: 15 External Control of System or Configuration Setting
 * BadSource: Environment Read data from an environment variable
 * GoodSource: A hardcoded string
 * Sinks:
 *    BadSink : Set the catalog name with the value of data
 * Flow Variant: 22 Control flow: Flow controlled by value of a public static variable. Sink functions are in a separate file from sources.
 *
 * */







public class CWE15_External_Control_of_System_or_Configuration_Setting__Environment_22b
{
    public String badSource() throws Throwable
    {
        String data = null;

        if (CWE15_External_Control_of_System_or_Configuration_Setting__Environment_22a.badPublicStatic)
        {
            /* get environment variable ADD */
            /* POTENTIAL FLAW: Read data from an environment variable */
            data = "bla";
        }
        return data;
    }

    /* goodG2B1() - use goodsource and badsink by setting the static variable to false instead of true */
    public String goodG2B1Source() throws Throwable
    {
        String data = "foo";

        return data;
    }

    /* goodG2B2() - use goodsource and badsink by reversing the blocks in the if in the sink function */
    public String goodG2B2Source() throws Throwable
    {
        String data = null;

        if (CWE15_External_Control_of_System_or_Configuration_Setting__Environment_22a.goodG2B2PublicStatic)
        {
            /* FIX: Use a hardcoded string */
            data = "foo";
        }

        return data;
    }
}