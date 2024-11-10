from faicons import icon_svg
import plotly.express as px

# Import data from shared.py
from shared import app_dir, df
import helper
import modelrunner
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

from functools import partial

from shiny import reactive
from shiny.express import input, render, ui
from shiny.ui import page_navbar
from shinywidgets import render_widget  

from ipyleaflet import Map, Heatmap, Marker, CircleMarker, MarkerCluster

ui.page_opts(
    title="Home Valuation Dashboard", 
    fillable=True,
    id="page"
)

ui.input_dark_mode(id="mode")

@reactive.effect
@reactive.event(input.make_light)
def _():
    ui.update_dark_mode("light")

@reactive.effect
@reactive.event(input.make_dark)
def _():
    ui.update_dark_mode("dark")

with ui.sidebar(title="Filter controls"):
    ui.input_slider("price", "Price", min=200000, max=2500000, step=1000, value = [200000,2500000], pre ="$")
    ui.input_slider("livingarea", "Living Area", min=0, max=20000, step=1000, value = [0,20000], post =" sqft")
    ui.input_checkbox_group(
        "homeType",
        "Home Type",
        ["SINGLE_FAMILY", "CONDO", "TOWNHOUSE", "MULTI_FAMILY"],
        selected=["SINGLE_FAMILY", "CONDO", "TOWNHOUSE", "MULTI_FAMILY"],
    )
    ui.input_checkbox_group(
        "bedrooms",
        "Number of Bedrooms",
        ["0", "1", "2", "3", "4+"],
        selected=["0", "1", "2", "3", "4+"],
    )
    ui.input_checkbox_group(
        "bathrooms",
        "Number of Bathrooms",
        ["0", "1", "2", "3", "4+"],
        selected=["0", "1", "2", "3", "4+"],
    )
    ui.input_checkbox_group(
        "halfbathrooms",
        "Number of Half Bathrooms",
        ["0", "1", "2", "3", "4+"],
        selected=["0", "1", "2", "3", "4+"],
    )
    ui.input_action_button("apply", "Apply Changes")


with ui.navset_pill(id="tab"):
    with ui.nav_panel("Data View"):
        "See Data Here"
        with ui.card(full_screen=True):
        # with ui.sidebar(title="Filter controls"):
        #     ui.input_slider("price", "Price", min=200000, max=2500000, step=1000, value = [500000,1000000], pre ="$")
        #     ui.input_checkbox_group(
        #         "homeType",
        #         "Home Type",
        #         ["SINGLE_FAMILY", "CONDO", "TOWNHOUSE", "MULTI_FAMILY"],
        #         selected=["SINGLE_FAMILY", "CONDO", "TOWNHOUSE", "MULTI_FAMILY"],
        #     )
        # with ui.layout_columns():
            # with ui.card(full_screen=True):
            #     ui.card_header("Housing Graphs")

                # @render.plot
                # def length_depth():
                #     return sns.scatterplot(
                #         data=filtered_df(),
                #         x="bill_length_mm",
                #         y="bill_depth_mm",
                #         hue="species",
                #     )

            # with ui.card(full_screen=True):
            #     ui.card_header("Housing Data")
            # with ui.layout_columns():  
            @render.data_frame
            def summary_statistics():
                cols = [
                    "price",
                    "livingArea",
                    "bedrooms",
                    "fullBathrooms",
                    "halfBathrooms",
                    "homeType",
                    "yearBuilt",
                    "city",
                    "propertyTaxRate"
                ]
                return render.DataGrid(filtered_df.get()[cols], filters=True)


    with ui.nav_panel("Data Visualization"):
        ui.input_slider("nbins", "Number of bins", 1, 100, 50)
        with ui.layout_column_wrap(width=1 / 2):
            with ui.card(full_screen=True):
                ui.card_header("Housing Map")
                @render_widget  
                def map():
                    california_center = [36, -119.7526]  # Latitude, Longitude
                    zoom_level = 6  # Good zoom level to see most of California

                    m = Map(
                        center=california_center,
                        zoom=zoom_level,
                        scroll_wheel_zoom=True
                    )

                    heatmap = Heatmap(
                        locations=coordinates.get(),
                        radius=20
                    )
                    m.add(heatmap)

                    markers = [CircleMarker(location=[lat, lon], radius=5) for lat, lon in coordinates.get()]
                    cluster = MarkerCluster(markers=markers)
                    m.add_layer(cluster)

                    return m

            with ui.card(full_screen=True):
                @render_widget  
                def priceplot():
                    # Determine the mode (light or dark)
                    mode = input.mode()

                    # Set the color for light/dark mode
                    color_fg = "black" if mode == "light" else "silver"

                    histogram = px.histogram(
                        data_frame=filtered_df.get(),
                        x="price",
                        nbins=input.nbins(),
                    ).update_layout(
                        title={"text": "Price Distribution", "x": 0.5},
                        # margin=dict(l=20, r=20, t=20, b=20),
                        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background for the plot
                        paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background for the paper
                        xaxis=dict(
                            title="Price (USD)",
                            tickcolor=color_fg,
                            linecolor=color_fg,
                            showgrid=False  # Hide gridlines
                        ),
                        yaxis=dict(
                            title="Count",
                            tickcolor=color_fg,
                            linecolor=color_fg,
                            showgrid=False  # Hide gridlines
                        ),
                        font=dict(color=color_fg),  # Font color for axis labels and title
                    )

                    return histogram

            with ui.card(full_screen=True):
                @render_widget  
                def livingareaplot():
                    # Determine the mode (light or dark)
                    mode = input.mode()

                    # Set the color for light/dark mode
                    color_fg = "black" if mode == "light" else "silver"

                    histogram = px.histogram(
                        data_frame=filtered_df.get(),
                        x="livingArea",
                        nbins=input.nbins(),
                    ).update_layout(
                        title={"text": "Living Area Distribution", "x": 0.5},
                        # margin=dict(l=20, r=20, t=20, b=20),
                        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background for the plot
                        paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background for the paper
                        xaxis=dict(
                            title="Living Area (Square Feet)",
                            tickcolor=color_fg,
                            linecolor=color_fg,
                            showgrid=False  # Hide gridlines
                        ),
                        yaxis=dict(
                            title="Count",
                            tickcolor=color_fg,
                            linecolor=color_fg,
                            showgrid=False  # Hide gridlines
                        ),
                        font=dict(color=color_fg),  # Font color for axis labels and title
                    )

                    return histogram

            with ui.card(full_screen=True):
                @render_widget  
                def yearBuiltplot():
                    # Determine the mode (light or dark)
                    mode = input.mode()

                    # Set the color for light/dark mode
                    color_fg = "black" if mode == "light" else "silver"

                    histogram = px.histogram(
                        data_frame=filtered_df.get(),
                        x="yearBuilt",
                        nbins=input.nbins(),
                    ).update_layout(
                        title={"text": "Year Built Distribution", "x": 0.5},
                        # margin=dict(l=20, r=20, t=20, b=20),
                        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background for the plot
                        paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background for the paper
                        xaxis=dict(
                            title="Year Built",
                            tickcolor=color_fg,
                            linecolor=color_fg,
                            showgrid=False  # Hide gridlines
                        ),
                        yaxis=dict(
                            title="Count",
                            tickcolor=color_fg,
                            linecolor=color_fg,
                            showgrid=False  # Hide gridlines
                        ),
                        font=dict(color=color_fg),  # Font color for axis labels and title
                    )

                    return histogram


    with ui.nav_panel("Get Model Prediction"):
        "Get model prediction here"
        with ui.layout_column_wrap():
            with ui.card():
                ui.input_numeric("lotSize_predict", "Lot Size (sqft)", 0, min=0, max=1000000)
                ui.input_numeric("livingArea_predict", "Living Area (sqft)", 0, min=0, max=1000000)
                ui.input_numeric("bedrooms_predict", "Bedrooms", 0, min=0, max=10, step=1)
                ui.input_numeric("fullBathrooms_predict", "Full Bathrooms", 0, min=0, max=10, step=1)
                ui.input_numeric("halfBathrooms_predict", "Half Bathrooms", 0, min=0, max=10, step=1)
                ui.input_numeric("yearBuilt_predict", "Year Built", 2020, min=1850, max=2024, step=1)
                ui.input_numeric("propertytaxrate_predict", "Property Tax Rate (%)", .71, min=0, max=5, step=.01)
            with ui.card():
                ui.input_text("address_predict", "Address", "")
                ui.input_text("zipcode_predict", "Zipcode", "")
                ui.input_text("city_predict", "City (in California)", "")
                ui.input_radio_buttons(  
                    "homeType_predict",  
                    "Home Type",  
                    {"SINGLE_FAMILY": "SINGLE FAMILY", "CONDO": "CONDO", "TOWNHOUSE": "TOWNHOUSE", "MULTI_FAMILY": "MULTI FAMILY", "APARTMENT": "APARTMENT", "HOME_TYPE_UNKNOWN": "HOME TYPE UNKNOWN", "MANUFACTURED": "MANUFACTURED"},  
                )
                ui.input_checkbox("hasGarage_predict", "Has Garage", False)
                ui.input_checkbox("hasAttachedProperty_predict", "Has Attached Property", False)
                ui.input_checkbox("hasView_predict", "Has View", False)
                ui.input_checkbox("hasHOA_predict", "Has HOA", False)

                with ui.panel_conditional("input.hasHOA_predict"):
                    ui.input_numeric("AnnualHOAFee_predict", "Annual HOA Fee ($)", 1000, min=0)

                ui.input_checkbox("hasHomeWarranty_predict", "Has Home Warranty", False)
        ui.input_action_button("predict", "Predict")

        "The property is predicted to be valued at: "

        @render.text(inline=True)
        @reactive.event(input.predict)
        async def prediction():
            with ui.Progress(min=1, max=5) as p:
                p.set(value=1, message="Predicting...", detail="This may take a while...")
                
                # Get current filter values
                selected_lotsize = input.lotSize_predict()
                selected_living_area = input.livingArea_predict()
                selected_home_types = input.homeType_predict()
                selected_bedrooms = input.bedrooms_predict()
                selected_bathrooms = input.fullBathrooms_predict()
                selected_halfbathrooms = input.halfBathrooms_predict()
                selected_yearbuilt = input.yearBuilt_predict()
                selected_propertytaxrate = input.propertytaxrate_predict()
                selected_address = input.address_predict()
                selected_zipcode = input.zipcode_predict()
                selected_city = input.city_predict()
                selected_hasgarage = input.hasGarage_predict()
                selected_hasattachedproperty = input.hasAttachedProperty_predict()
                selected_hasview = input.hasView_predict()
                selected_hasHOA = input.hasHOA_predict()
                if selected_hasHOA:
                    selected_HOAFee = input.AnnualHOAFee_predict()
                selected_hashomewarranty = input.hasHomeWarranty_predict()

                df = create_property_dataframe(
                    lotSize=selected_lotsize,
                    livingArea=selected_living_area,
                    homeType=selected_home_types,
                    bedrooms=selected_bedrooms,
                    bathrooms=selected_bathrooms,
                    halfBathrooms=selected_halfbathrooms,
                    yearBuilt=selected_yearbuilt,
                    propertyTaxRate=selected_propertytaxrate,
                    address=selected_address,
                    zipcode=selected_zipcode,
                    city=selected_city,
                    hasGarage=selected_hasgarage,
                    hasAttachedProperty=selected_hasattachedproperty,
                    hasView=selected_hasview,
                    hasHOA=selected_hasHOA,
                    AnnualHOAFee=selected_HOAFee if selected_hasHOA else None,
                    hasHomeWarranty=selected_hashomewarranty
                )

                p.set(value=2, message="Predicting...", detail="Finding Address...")
                lat, lon = helper.geocode_address(selected_address, selected_city, selected_zipcode)

                p.set(value=3, message="Predicting...", detail="Finding Nearby Points of Interest...")
                nearest_features_distance_df = helper.findNearestWithIncreasingRadius(lat, lon)
                nearest_features_distance_only_df = {key: value['distance'] for key, value in nearest_features_distance_df.items()}
                df = pd.merge(df, pd.DataFrame([nearest_features_distance_only_df]), left_index=True, right_index=True, how='inner')

                p.set(value=4, message="Predicting...", detail="Finding Climate Data...")
                city_climate_df = helper.get_city_climate_data(selected_city)
                df = pd.merge(df, pd.DataFrame([city_climate_df]), on='city', how='left')
                df.drop(["city", "latitude", "longitude", "address", "zipcode"], axis=1, inplace=True)
                df = df.astype({
                    'lotSize': 'float64',
                    'livingArea': 'float64',
                    'AnnualHOAFee': 'float64',
                    'propertyTaxRate': 'float64',
                    'elevation': 'float64',
                    'mean_daily_high_temperature': 'float64',
                    'mean_daily_low_temperature': 'float64',
                    'rainfall': 'float64',
                    'population_density': 'float64',
                    'park': 'float64',
                    'motorway': 'float64',
                    'supermarket': 'float64',
                    'Costco': 'float64',
                    'hasGarage': 'bool',
                    'hasAttachedProperty': 'bool',
                    'hasView': 'bool',
                    'hasHOA': 'bool',
                    'hasHomeWarranty': 'bool',
                    'fullBathrooms': 'Int64',
                    'halfBathrooms': 'Int64',
                    'bedrooms': 'Int64',
                    'homeType_SINGLE_FAMILY': 'int8',
                    'homeType_TOWNHOUSE': 'int8',
                    'homeType_MANUFACTURED': 'int8',
                    'homeType_CONDO': 'int8',
                    'homeType_APARTMENT': 'int8',
                    'homeType_HOME_TYPE_UNKNOWN': 'int8',
                    'homeType_MULTI_FAMILY': 'int8',
                    'yearBuilt': 'Int64'
                })
                df_dummies = pd.get_dummies(df)

                p.set(value=5, message="Predicting...", detail="Making Prediction...")
                model, input_features = modelrunner.load_xgboost_model("../models/", "avm_model")
                prediction = modelrunner.make_prediction(model, input_features, df_dummies, required_features=input_features)
                prediction_value = prediction.values[0]
                prediction_value_str = f"${prediction_value:,.2f}"
                print(prediction_value_str)
                return str(prediction_value_str)


    with ui.nav_panel("Train New Models"):
        with ui.layout_column_wrap():
            with ui.card():
                ui.card_header("Features to train a new model on")
                ui.input_checkbox("lotSize_option", "Lot Size", True)
                ui.input_checkbox("livingArea_option", "Living Area", True)
                ui.input_checkbox("hasGarage_option", "has Garage", True)
                ui.input_checkbox("hasAttachedProperty_option", "has Attached Property", True)
                ui.input_checkbox("hasView_option", "has View", True)
                ui.input_checkbox("hasHOA_option", "has HOA", True)
                ui.input_checkbox("AnnualHOAFee_option", "Annual HOA Fee", True)
                ui.input_checkbox("fullBathrooms_option", "Full Bathrooms", True)
                ui.input_checkbox("halfBathrooms_option", "Half Bathrooms", True)
                ui.input_checkbox("bedrooms_option", "Bedrooms", True)
                ui.input_checkbox("city_option", "City", False)
                ui.input_checkbox("homeType_option", "Home Type", True)
                ui.input_checkbox("propertyTaxRate_option", "Property Tax Rate", True)
                ui.input_checkbox("yearBuilt_option", "Year Built", True)
                ui.input_checkbox("hasHomeWarranty_option", "has Home Warranty", True)
                ui.input_checkbox("latitude_option", "Latitude", False)
                ui.input_checkbox("longitude_option", "Longitude", False)
                ui.input_checkbox("elevation_option", "Elevation", True)
                ui.input_checkbox("mean_daily_high_temperature_option", "Mean daily high temperature", True)
                ui.input_checkbox("mean_daily_low_temperature_option", "Mean daily low temperature", True)
                ui.input_checkbox("rainfall_option", "Rainfall", True)
                ui.input_checkbox("population_density_option", "Population Density", True)
                ui.input_checkbox("park_option", "Distance from park", True)
                ui.input_checkbox("motorway_option", "Distance from Motorway/Highway", True)
                ui.input_checkbox("supermarket_option", "Distance from supermarket", True)
                ui.input_checkbox("Costco_option", "Distance from Costco", True)
                ui.input_action_button("train", "Train and Evaluate new model")
            with ui.card():
                ui.card_header("Model Evaluation")
                @render.text
                def rsquared_render():
                    return f"R Squared: {r_squared.get():.4f}"

                @render.text
                def rmse_render():
                    return f"Root mean squared error: ${rmse.get():,.2f}"
                
                @render_widget
                @reactive.event(input.train)
                def train_model():
                    with ui.Progress(min=1, max=6) as p:
                        p.set(value=1, message="Training...", detail="This may take a while...")

                        types = {
                            'lotSize': 'float64',
                            'livingArea': 'float64',
                            'AnnualHOAFee': 'float64',
                            'propertyTaxRate': 'float64',
                            'elevation': 'float64',
                            'mean_daily_high_temperature': 'float64',
                            'mean_daily_low_temperature': 'float64',
                            'rainfall': 'float64',
                            'population_density': 'float64',
                            'park': 'float64',
                            'motorway': 'float64',
                            'supermarket': 'float64',
                            'Costco': 'float64',
                            'hasGarage': 'bool',
                            'hasAttachedProperty': 'bool',
                            'hasView': 'bool',
                            'hasHOA': 'bool',
                            'hasHomeWarranty': 'bool',
                            'fullBathrooms': 'Int64',
                            'halfBathrooms': 'Int64',
                            'bedrooms': 'Int64',
                            'homeType': 'category',
                            'yearBuilt': 'Int64',
                            'city': 'category'
                        }

                        p.set(value=2, message="Importing Data...", detail="This may take a while...")

                        df = pd.read_csv("./housing.csv", dtype=types)

                        p.set(value=3, message="Filtering Data...", detail="This may take a while...")
                        
                        selected_features = []
                        # I wish there was a better way lol
                        if input.lotSize_option():
                            selected_features.append("lotSize")
                        if input.livingArea_option():
                            selected_features.append("livingArea")
                        if input.hasGarage_option():
                            selected_features.append("hasGarage")
                        if input.hasAttachedProperty_option():
                            selected_features.append("hasAttachedProperty")
                        if input.hasView_option():
                            selected_features.append("hasView")
                        if input.hasHOA_option():
                            selected_features.append("hasHOA")
                        if input.AnnualHOAFee_option():
                            selected_features.append("AnnualHOAFee")
                        if input.fullBathrooms_option():
                            selected_features.append("fullBathrooms")
                        if input.halfBathrooms_option():
                            selected_features.append("halfBathrooms")
                        if input.bedrooms_option():
                            selected_features.append("bedrooms")
                        if input.city_option():
                            selected_features.append("city")
                        if input.homeType_option():
                            selected_features.append("homeType")
                        if input.propertyTaxRate_option():
                            selected_features.append("propertyTaxRate")
                        if input.yearBuilt_option():
                            selected_features.append("yearBuilt")
                        if input.hasHomeWarranty_option():
                            selected_features.append("hasHomeWarranty")
                        if input.latitude_option():
                            selected_features.append("latitude")
                        if input.longitude_option():
                            selected_features.append("longitude")
                        if input.elevation_option():
                            selected_features.append("elevation")
                        if input.mean_daily_high_temperature_option():
                            selected_features.append("mean_daily_high_temperature")
                        if input.mean_daily_low_temperature_option():
                            selected_features.append("mean_daily_low_temperature")
                        if input.rainfall_option():
                            selected_features.append("rainfall")
                        if input.population_density_option():
                            selected_features.append("population_density")
                        if input.park_option():
                            selected_features.append("park")
                        if input.motorway_option():
                            selected_features.append("motorway")
                        if input.supermarket_option():
                            selected_features.append("supermarket")
                        if input.Costco_option():
                            selected_features.append("Costco")
                        
                        selected_features.append("price")

                        cut_df = df.loc[:, selected_features]

                        df_dummies = pd.get_dummies(cut_df)
                        
                        X = df_dummies.drop("price", axis=1)
                        y= df_dummies["price"]

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

                        p.set(value=4, message="Fitting Data to Model...", detail="This may take a while...")

                        model = xgb.XGBRegressor(n_estimators=1000)

                        model.fit(X_train, y_train)

                        p.set(value=5, message="Calculating metrics...", detail="This may take a while...")

                        r_squared.set(model.score(X_test, y_test))

                        y_pred = model.predict(X_test)

                        rmse.set(root_mean_squared_error(y_test, y_pred))

                        residuals = y_test - y_pred

                        p.set(value=6, message="Plotting Residuals...", detail="This may take a while...")

                        # Determine the mode (light or dark)
                        mode = input.mode()

                        # Set the color for light/dark mode
                        color_fg = "black" if mode == "light" else "silver"

                        scatterplot = px.scatter(
                            x=model.predict(X_test),
                            y=residuals,
                        ).add_hline(
                            y=0, 
                            line_width=3, 
                            line_dash="dash", 
                            line_color="red"
                        ).update_layout(
                            title={"text": "Model Residuals"},
                            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background for the plot
                            paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background for the paper
                            xaxis=dict(
                                title="Predicted Value ($)",
                                tickcolor=color_fg,
                                linecolor=color_fg,
                                showgrid=False  # Hide gridlines
                            ),
                            yaxis=dict(
                                title="Error ($)",
                                tickcolor=color_fg,
                                linecolor=color_fg,
                                showgrid=False  # Hide gridlines
                            ),
                            font=dict(color=color_fg),  # Font color for axis labels and title
                        )

                        return scatterplot


filtered_df = reactive.Value(df)
coordinates = reactive.Value(list(zip(df["latitude"], df["longitude"])))
r_squared = reactive.Value()
rmse = reactive.Value()

@render.ui
@reactive.event(input.apply)
def compute():
    with ui.Progress(min=1, max=15) as p:
        p.set(message="Filtering data...", detail="This may take a while...")
        
        # Get current filter values
        min_price, max_price = input.price()
        min_livingarea, max_livingarea = input.livingarea()
        home_types = input.homeType()
        selected_bedrooms = input.bedrooms()
        selected_bathrooms = input.bathrooms()
        selected_halfbathrooms = input.halfbathrooms()
        # Apply filters
        filt = df.copy()
        if home_types:
            filt = filt[filt["homeType"].isin(home_types)]

        if selected_bedrooms:
            conditions = []
            for bed in selected_bedrooms:
                if bed == "4+":
                    conditions.append(filt["bedrooms"] >= 4)
                else:
                    conditions.append(filt["bedrooms"] == int(bed))
            # Combine all conditions with OR (|)
            if conditions:
                filt = filt[pd.concat(conditions, axis=1).any(axis=1)]

        if selected_bathrooms:
            conditions = []
            for bath in selected_bathrooms:
                if bath == "4+":
                    conditions.append(filt["fullBathrooms"] >= 4)
                else:
                    conditions.append(filt["fullBathrooms"] == int(bath))
            # Combine all conditions with OR (|)
            if conditions:
                filt = filt[pd.concat(conditions, axis=1).any(axis=1)]

        if selected_halfbathrooms:
            conditions = []
            for bath in selected_halfbathrooms:
                if bath == "4+":
                    conditions.append(filt["halfBathrooms"] >= 4)
                else:
                    conditions.append(filt["halfBathrooms"] == int(bath))
            # Combine all conditions with OR (|)
            if conditions:
                filt = filt[pd.concat(conditions, axis=1).any(axis=1)]

        
        filt = filt.loc[(filt["price"] >= min_price) & (filt["price"] <= max_price)]
        filt = filt.loc[(filt["livingArea"] >= min_livingarea) & (filt["livingArea"] <= max_livingarea)]

        # Update the reactive value
        filtered_df.set(filt)

        coordinates.set(list(zip(filt["latitude"], filt["longitude"])))



def create_property_dataframe(
    lotSize,
    livingArea,
    homeType,
    bedrooms,
    bathrooms,
    halfBathrooms,
    yearBuilt,
    propertyTaxRate,
    address,
    zipcode,
    city,
    hasGarage,
    hasAttachedProperty,
    hasView,
    hasHOA,
    hasHomeWarranty,
    AnnualHOAFee=None
):
    """
    Creates a pandas DataFrame from property information.
    
    Args:
        All relevant property features
        HOAFee (float, optional): Only required if hasHOA is True
        
    Returns:
        pandas.DataFrame: Single row DataFrame with all property features
    """
    
    # Create dictionary with all properties
    property_dict = {
        'lotSize': lotSize,
        'livingArea': livingArea,
        'homeType_SINGLE_FAMILY': 0,
        'homeType_TOWNHOUSE': 0,
        'homeType_MANUFACTURED': 0,
        'homeType_CONDO': 0,
        'homeType_APARTMENT': 0,
        'homeType_HOME_TYPE_UNKNOWN': 0,
        'homeType_MULTI_FAMILY': 0,
        'bedrooms': bedrooms,
        'fullBathrooms': bathrooms,
        'halfBathrooms': halfBathrooms,
        'yearBuilt': yearBuilt,
        'propertyTaxRate': propertyTaxRate,
        'address': address,
        'zipcode': zipcode,
        'city': city,
        'hasGarage': hasGarage,
        'hasAttachedProperty': hasAttachedProperty,
        'hasView': hasView,
        'hasHOA': hasHOA,
        'AnnualHOAFee': AnnualHOAFee if hasHOA else 0,
        'hasHomeWarranty': hasHomeWarranty
    }
    
    # Create DataFrame with a single row
    df = pd.DataFrame([property_dict])
    
    column_name = f"homeType_{homeType}"

    if column_name in df.columns:
        df[column_name] = 1
    
    return df