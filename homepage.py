import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pydeck as pdk

# Set the page configuration
st.set_page_config(page_title="Impact of Social Determinants on Health Outcomes", layout="wide")

# Load data for Income vs Health page
@st.cache_data
def load_income_data():
    data = pd.read_csv('./Filtered_Dataset/healthandincome.csv')  # Update the path to your data file
    return data

# Load data for Education vs Health page
def load_education_data():
    education_data_path = './Filtered_Dataset/allyear.csv'
    binge_drinking_data_path = './Filtered_Dataset/Filtered_Data.csv'
    
    education_data = pd.read_csv(education_data_path)
    health_data = pd.read_csv(binge_drinking_data_path)
    return education_data, health_data

df_income = load_income_data()
df_education, df_health = load_education_data()

# Define navigation options
pages = {
    "Home": "home",
    "Income vs Health": "income_vs_health",
    "Education vs Health": "education_vs_health",
    "Housing vs Health": "housing_vs_health",
}

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", list(pages.keys()))

# Home page content
if page == "Home":
    st.title("Impact of Social Determinants on Health Outcomes")
    st.write("""
    The purpose of this specific project is to display the overall linkage between social determinants of health, like low income, low educational attainment, and poor housing, can all contribute to an individual's overall health status. While this research can be performed on a national level, we will be specifically focused on the linkage found within different California communities. Overall, the purpose of understanding the relationships between these two different factors, SDOH and overall health, is crucial for the California government to understand as it can lead to the development of more equitable health policies within the community and overall state. In doing so, California would help diminish, and potentially eliminate, the overall health disparities that are currently present within the health sector and improve the overall health of its multiple counties. Not only that, but the California government would also be able to eliminate needless deaths that occur due to specific individual factors within SDOH.

    ### Features:
    - **Interactive Map:** Visualize health outcomes by region.
    - **Correlation Analysis:** Explore relationships between income and disease prevalence through scatter plots and bar charts.
    - **Temporal Trends:** Analyze changes over time with line charts.
    
    Navigate through the different pages using the sidebar to explore the features of this project.
    """)

# Income vs Health page content
elif page == "Income vs Health":
    st.title("Health Outcomes and Social Determinants Analysis for Income and Health")
    st.header("Income vs Health")
    question_text_map = st.selectbox('Select a Short Question Text for the Map', df_income['Short_Question_Text'].unique(), key='map_question')
    map_df = df_income[df_income['Short_Question_Text'] == question_text_map].dropna(subset=['GeoLocation', 'Data_Value', 'Median_Income'])
    map_df[['lat', 'lon']] = map_df['GeoLocation'].str.extract(r'\(([^,]+), ([^,]+)\)')
    map_df['lat'] = map_df['lat'].astype(float)
    map_df['lon'] = map_df['lon'].astype(float)

    fig = px.scatter_mapbox(
        map_df, lat="lat", lon="lon", hover_name="CityName",
        hover_data={"lat": True, "lon": True, "Data_Value": True, "Median_Income": True},
        color="Data_Value", zoom=5, height=500
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig)

    st.header("Correlation Analysis")
    question_text = st.selectbox('Select a Short Question Text', df_income['Short_Question_Text'].unique(), key='correlation_question')
    filtered_df = df_income[df_income['Short_Question_Text'] == question_text]
    x_axis = st.selectbox('Select X-axis variable', ['Median_Income'], key='correlation_x_axis')
    y_axis = 'Data_Value'

    # Hexbin plot for correlation analysis
    fig, ax = plt.subplots(figsize=(10, 8))
    hb = ax.hexbin(filtered_df[x_axis], filtered_df[y_axis], gridsize=30, cmap='Blues', mincnt=1)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Count')

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f'Hexbin plot between {x_axis} and {y_axis} for "{question_text}"')
    st.pyplot(fig)

    st.header("Bar Charts Comparing Different Regions or Demographics")
    region = st.selectbox('Select a region', df_income['CityName'].unique(), key='bar_region')
    bar_data = df_income[df_income['CityName'] == region]

    # Display median income for the selected region
    median_income = bar_data['Median_Income'].iloc[0]
    st.write(f"**Median Income for {region}:** ${median_income}")

    fig, ax = plt.subplots()
    sns.barplot(data=bar_data, x='Short_Question_Text', y='Data_Value', ax=ax)
    ax.set_title(f'Bar chart of health outcomes in {region}')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig)

    st.header("Example Narratives")
    st.markdown("""
    ### Income and Health Outcomes

    1. **Life Expectancy**: Higher income levels are consistently associated with longer life expectancy. Wealthier individuals and communities typically have better access to healthcare services, healthier living conditions, and lower levels of chronic stress, all contributing to increased longevity.

    2. **Infant Mortality**: Income levels also impact infant mortality rates. Higher-income families can afford better prenatal and postnatal care, reducing the risk of complications during pregnancy and childbirth. In contrast, low-income families often face barriers to accessing quality healthcare, leading to higher infant mortality rates.

    3. **Chronic Diseases**: Lower income is linked to higher prevalence of chronic diseases such as diabetes, heart disease, and obesity. This is often due to factors like limited access to healthy foods, lower levels of physical activity, and increased stress.

    4. **Mental Health**: Economic instability and low income are significant risk factors for mental health disorders, including depression and anxiety. Financial stress and limited access to mental health services exacerbate these issues.
    """)

    st.markdown("""

    """)

# Education vs Health page content
elif page == "Education vs Health":
    st.title("California Education and Health Visualization")

    # Bachelor's Degree Data
    st.subheader('Bachelor\'s Degree Holders by County')
    bachelors_data = df_education.loc[df_education['Label (Grouping)'].str.contains("Bachelor's degree")].iloc[0]
    bachelors_data_total = bachelors_data[1::2]  # Extract total number columns (skip percentage columns)

    # Convert the total bachelors data to a DataFrame with proper coordinates
    bachelors_data_df = pd.DataFrame({
        'Coordinates': bachelors_data_total.index,
        'Total_Bachelors': bachelors_data_total.values
    })

    # Separate coordinates into latitude and longitude for plotting
    bachelors_data_df['Coordinates'] = bachelors_data_df['Coordinates'].apply(lambda x: eval(x))
    bachelors_data_df[['Longitude', 'Latitude']] = pd.DataFrame(bachelors_data_df['Coordinates'].tolist(), index=bachelors_data_df.index)
    bachelors_data_df = bachelors_data_df.drop(columns=['Coordinates'])

    # Clean and convert total bachelors to integers
    bachelors_data_df['Total_Bachelors'] = bachelors_data_df['Total_Bachelors'].str.replace(',', '').astype(int)

    # Function to create map layers
    def create_map_layer(data, color, tooltip_text):
        return pdk.Layer(
            'ScatterplotLayer',
            data=data,
            get_position='[Longitude, Latitude]',
            get_radius=5000,
            get_fill_color=color,
            pickable=True,
            tooltip={"text": tooltip_text}
        )

    bachelor_layer = create_map_layer(
        bachelors_data_df,
        '[200, 30, 0, 160]',
        "Total Bachelors: {Total_Bachelors}"
    )

    # List of health indicators to visualize
    health_indicators = [
        'Binge Drinking',
    ]

    # Create visualizations for each health indicator
    for indicator in health_indicators:
        st.subheader(f'{indicator} vs Bachelor\'s Degree Holders')
        
        # Filter health data for the selected indicator
        health_data_filtered = df_health[df_health['Short_Question_Text'] == indicator]
        health_data_filtered = health_data_filtered[['GeoLocation', 'Data_Value']]
        
        # Drop rows with NaN values in GeoLocation
        health_data_filtered = health_data_filtered.dropna(subset=['GeoLocation'])
        
        # Convert GeoLocation to tuples of floats for plotting
        health_data_filtered['GeoLocation'] = health_data_filtered['GeoLocation'].apply(
        lambda x: tuple(map(float, x.strip('()').split(','))) if isinstance(x, str) else x
        )
        
        # Split GeoLocation into Latitude and Longitude
        health_data_filtered[['Latitude', 'Longitude']] = pd.DataFrame(health_data_filtered['GeoLocation'].tolist(), index=health_data_filtered.index)
        
        # Ensure Data_Value is numeric
        health_data_filtered['Data_Value'] = pd.to_numeric(health_data_filtered['Data_Value'], errors='coerce')
        
        # Create health layer
        health_layer = create_map_layer(
            health_data_filtered,
            '[0, 100, 200, 160]',
            f"{indicator} Percentage: {{Data_Value}}"
        )
        
        # Deck.gl Map for the current health indicator
        view_state = pdk.ViewState(
            longitude=-119.4179,
            latitude=36.7783,
            zoom=5,
            pitch=0
        )
        
        r = pdk.Deck(layers=[bachelor_layer, health_layer], initial_view_state=view_state)
        st.pydeck_chart(r)

    st.markdown("""
    ### Education and Binge Drinking in California

    This interactive map visualizes the relationship between educational attainment (specifically, bachelor's degree holders) and binge drinking across California counties.

    #### Visualization Description

    The map contains two types of markers:
    - **Red markers**: Represent the number of bachelor's degree holders in each county.
    - **Blue markers**: Represent the prevalence of binge drinking in each county.

    #### Key Insights

    1. **Education and Binge Drinking**:
       - There seems to be a correlation between bachelor's degree holders and binge drinking. Urban areas like the San Francisco Bay Area and Los Angeles, which have many bachelor's degree holders, also show significant binge drinking.

    2. **Rural vs. Urban Trends**:
       - Rural areas in California, with fewer bachelor's degree holders, also show fewer instances of binge drinking. This could be due to less social activity and different lifestyle choices compared to urban areas.

    """)

# Housing vs Health page content
elif page == "Housing vs Health":
    st.title("Housing Quality and Health Outcomes in California")

    # Load the dataset for housing and health
    file_path = './Filtered_Dataset/healthhouse.csv'  # Update the path if needed
    data = pd.read_csv(file_path)

    # Ensure the relevant columns are treated as numeric
    data['Lacking_plumbing'] = pd.to_numeric(data['Lacking_plumbing'], errors='coerce')
    data['Lacking_kitchen'] = pd.to_numeric(data['Lacking_kitchen'], errors='coerce')
    data['No_telephone_service'] = pd.to_numeric(data['No_telephone_service'], errors='coerce')
    data['Data_Value'] = pd.to_numeric(data['Data_Value'], errors='coerce')

    # Health Outcomes options
    health_outcomes = data['Short_Question_Text'].unique()
    selected_health_outcome = st.sidebar.selectbox('Select Health Outcome', health_outcomes)

    # Filter data based on selected health outcome
    map_data = data[data['Short_Question_Text'] == selected_health_outcome].copy()

    # Prepare geolocation data for mapping
    map_data['Geolocation'] = map_data['Geolocation'].apply(lambda x: x.strip('POINT ()').split())
    map_data['latitude'] = map_data['Geolocation'].apply(lambda x: float(x[1]))
    map_data['longitude'] = map_data['Geolocation'].apply(lambda x: float(x[0]))

    # Map visualization for a broader overview
    st.header('Map Visualization of Health Outcomes by Region')

    fig = px.scatter_mapbox(map_data, lat='latitude', lon='longitude', color='Data_Value',
                            hover_name='LocationName', hover_data=['Lacking_plumbing', 'Lacking_kitchen', 'No_telephone_service'],
                            title=f'{selected_health_outcome} by Region',
                            color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=5)

    fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig)

    # Lacking Kitchen vs Obesity - Bar Chart
    st.header('Lacking Kitchen vs Obesity')

    obesity_data = data[data['Short_Question_Text'] == 'Obesity']
    bins = [0, 1, 2, 3, 4]
    labels = ['0-1%', '1-2%', '2-3%', '3-4%']
    obesity_data['Lacking_kitchen_bin'] = pd.cut(obesity_data['Lacking_kitchen'], bins=bins, labels=labels, include_lowest=True)
    fig1 = px.bar(obesity_data, x='Lacking_kitchen_bin', y='Data_Value',
                  title='Lacking Kitchen vs Obesity',
                  labels={'Lacking_kitchen_bin': 'Lacking Kitchen (%)', 'Data_Value': 'Obesity Prevalence (%)'},
                  color='Lacking_kitchen_bin', barmode='group')
    st.plotly_chart(fig1)

    # Lacking Plumbing vs Chronic Kidney Disease - Bar Chart
    st.header('Lacking Plumbing vs Chronic Kidney Disease')

    ckd_data = data[data['Short_Question_Text'] == 'Chronic Kidney Disease']
    ckd_data['Lacking_plumbing_bin'] = pd.cut(ckd_data['Lacking_plumbing'], bins=bins, labels=labels, include_lowest=True)
    fig2 = px.bar(ckd_data, x='Lacking_plumbing_bin', y='Data_Value',
                  title='Lacking Plumbing vs Chronic Kidney Disease',
                  labels={'Lacking_plumbing_bin': 'Lacking Plumbing (%)', 'Data_Value': 'Chronic Kidney Disease Prevalence (%)'},
                  color='Lacking_plumbing_bin', barmode='group')
    st.plotly_chart(fig2)


